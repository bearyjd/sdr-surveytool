-- storage/sql/agent_boundary.sql
--
-- The Part 4 agent's database boundary. Idempotent: safe to apply again. A
-- reinstall drops and recreates the view and both functions, so no grant
-- added to them since the last install survives it.
-- Applied only by storage.agent_boundary (an admin CLI), as one transaction,
-- never at ingest or agent startup. {{agent_role}} and {{owner_role}} are
-- substituted by storage.agent_boundary.render_agent_boundary_sql after
-- validating both against ^[a-z_][a-z0-9_]{0,62}$.
--
-- The agent role gets NO privilege on public.survey_records. It reads the
-- security_barrier view public.agent_pending_unknown and writes only through
-- public.classify_unknown, a SECURITY DEFINER function. Both are owned by a
-- NOLOGIN owner role that holds exactly SELECT and UPDATE (metadata) on the
-- table, so a superuser is never the definer.

SET LOCAL search_path = pg_catalog, pg_temp;

-- Roles. Operators create the login role themselves:
--   CREATE ROLE <login> LOGIN PASSWORD '<from your secret store>' IN ROLE {{agent_role}};
DO $roles$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = '{{agent_role}}') THEN
        CREATE ROLE {{agent_role}} NOLOGIN;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = '{{owner_role}}') THEN
        CREATE ROLE {{owner_role}} NOLOGIN;
    END IF;
END
$roles$;
ALTER ROLE {{agent_role}} NOLOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS;
ALTER ROLE {{owner_role}} NOLOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS;

-- Revokes. CREATE on public is revoked explicitly: clusters upgraded from
-- before PostgreSQL 15 keep the old PUBLIC CREATE default. TEMPORARY is
-- revoked so no role can define a leaky pg_temp function to probe the view.
REVOKE ALL ON TABLE public.survey_records FROM PUBLIC;
REVOKE ALL ON TABLE public.survey_records FROM {{agent_role}};
REVOKE ALL ON TABLE public.survey_records FROM {{owner_role}};
REVOKE CREATE ON SCHEMA public FROM PUBLIC;
DO $temp$
BEGIN
    EXECUTE pg_catalog.format('REVOKE TEMPORARY ON DATABASE %I FROM PUBLIC',
                              pg_catalog.current_database());
END
$temp$;

-- The owner role holds exactly what the view and classify_unknown need.
GRANT SELECT, UPDATE (metadata) ON TABLE public.survey_records TO {{owner_role}};

-- Start the boundary objects afresh: CREATE OR REPLACE would keep their
-- ACLs, and a view's column types cannot change in place. No CASCADE: an
-- object someone built on top of them makes the install fail loudly.
DROP VIEW IF EXISTS public.agent_pending_unknown;
DROP FUNCTION IF EXISTS public.classify_unknown(integer, text, text, double precision, text);
DROP FUNCTION IF EXISTS public.agent_is_pending_unknown(text, json);

-- The pending predicate, defined once and used by both the view and
-- classify_unknown. A SQL-standard body is parsed now, so its operators are
-- bound to pg_catalog at creation instead of resolved through the caller's
-- search_path at run time; it is still inlined into the view's plan.
CREATE FUNCTION public.agent_is_pending_unknown(p_modality text, p_metadata json)
    RETURNS boolean
    LANGUAGE sql
    IMMUTABLE
    RETURN p_modality = 'unknown'
       AND coalesce(p_metadata ->> 'classification_status', 'unclassified') = 'unclassified';
ALTER FUNCTION public.agent_is_pending_unknown(text, json) OWNER TO {{owner_role}};
REVOKE ALL ON FUNCTION public.agent_is_pending_unknown(text, json) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION public.agent_is_pending_unknown(text, json) TO {{agent_role}};

-- The agent's only read path: pending unknown rows, and only the columns the
-- agent needs (no location, no survey or operator ID, no other modality).
-- A pending row may have no snippet (iq_snippet_path NULL) when ingest
-- rejected it or capture dropped it; the two flags say which, so the agent
-- can close such rows out instead of leaving them pending forever.
-- security_barrier makes PostgreSQL apply this WHERE clause before any
-- non-leakproof condition a caller adds, so a caller's function can never see
-- a hidden row.
CREATE VIEW public.agent_pending_unknown
    WITH (security_barrier = true) AS
SELECT r.id,
       r.metadata ->> 'iq_snippet_path' AS iq_snippet_path,
       (r.metadata ->> 'sample_rate')::double precision AS sample_rate,
       (r.identifier ->> 'center_freq')::double precision AS center_freq,
       (r.signal ->> 'peak_power')::double precision AS peak_power,
       (r.metadata ->> 'snippet_duration_ms')::integer AS snippet_duration_ms,
       r.metadata -> 'quality_flags' ->> 'snippet_rejected' AS snippet_rejected,
       r.metadata -> 'quality_flags' ->> 'snippet_dropped' AS snippet_dropped
  FROM public.survey_records AS r
 WHERE public.agent_is_pending_unknown(r.modality, r.metadata);
ALTER VIEW public.agent_pending_unknown OWNER TO {{owner_role}};
REVOKE ALL ON TABLE public.agent_pending_unknown FROM PUBLIC;
GRANT SELECT ON TABLE public.agent_pending_unknown TO {{agent_role}};

-- The agent's only write path. SECURITY DEFINER runs as {{owner_role}};
-- search_path pins pg_catalog first and pg_temp explicitly last (unlisted,
-- pg_temp is searched first). Every object reference is schema-qualified.
-- SQLSTATE 22023 (invalid_parameter_value): bad arguments.
-- SQLSTATE P0002 (no_data_found): the row is not, or no longer, a pending
-- unknown row -- e.g. a human tagged it while the LLM call was in flight.
CREATE FUNCTION public.classify_unknown(
    p_record_id integer,
    p_status text,
    p_tag text,
    p_confidence double precision,
    p_reasoning text
)
    RETURNS void
    LANGUAGE plpgsql
    SECURITY DEFINER
    SET search_path = pg_catalog, pg_temp
AS $fn$
DECLARE
    v_id integer;
BEGIN
    IF p_status IS NULL OR p_status NOT IN ('auto_classified', 'needs_review') THEN
        RAISE EXCEPTION 'classify_unknown: invalid status %', p_status
            USING ERRCODE = 'invalid_parameter_value';
    END IF;
    IF p_confidence IS NULL OR NOT (p_confidence >= 0 AND p_confidence <= 1) THEN
        RAISE EXCEPTION 'classify_unknown: confidence % is outside [0, 1]', p_confidence
            USING ERRCODE = 'invalid_parameter_value';
    END IF;
    IF p_tag IS NULL THEN
        IF p_status <> 'needs_review' THEN
            RAISE EXCEPTION 'classify_unknown: a NULL tag requires status needs_review'
                USING ERRCODE = 'invalid_parameter_value';
        END IF;
    ELSIF p_tag !~ '^[a-z0-9][a-z0-9_.:-]{0,63}$' THEN
        RAISE EXCEPTION 'classify_unknown: invalid tag %', p_tag
            USING ERRCODE = 'invalid_parameter_value';
    END IF;
    IF p_reasoning IS NULL OR length(p_reasoning) > 4000 THEN
        RAISE EXCEPTION 'classify_unknown: reasoning must be non-NULL and at most 4000 characters'
            USING ERRCODE = 'invalid_parameter_value';
    END IF;

    -- Atomic: the pending check and the write are one statement. A concurrent
    -- writer's committed change is re-checked against the WHERE clause under
    -- READ COMMITTED, so a human tag always wins.
    UPDATE public.survey_records AS r
       SET metadata = (r.metadata::jsonb || jsonb_build_object(
               'classification_status', p_status,
               'tag', p_tag,
               'confidence', p_confidence,
               'reasoning', p_reasoning))::json
     WHERE r.id = p_record_id
       AND public.agent_is_pending_unknown(r.modality, r.metadata)
    RETURNING r.id INTO v_id;
    IF v_id IS NULL THEN
        RAISE EXCEPTION 'classify_unknown: record % is not a pending unknown record', p_record_id
            USING ERRCODE = 'no_data_found';
    END IF;
END
$fn$;
ALTER FUNCTION public.classify_unknown(integer, text, text, double precision, text)
    OWNER TO {{owner_role}};
REVOKE ALL ON FUNCTION public.classify_unknown(integer, text, text, double precision, text)
    FROM PUBLIC;
GRANT EXECUTE ON FUNCTION public.classify_unknown(integer, text, text, double precision, text)
    TO {{agent_role}};
