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

-- Roles. Operators create the login role themselves and set its password
-- with psql's \password (agent/README.md), never in SQL text. An
-- existing role is reused only if it cannot log in and owns nothing but the
-- boundary objects below: otherwise the boundary would inherit its powers.
DO $existing$
DECLARE
    v_role text;
BEGIN
    FOREACH v_role IN ARRAY ARRAY['{{agent_role}}', '{{owner_role}}'] LOOP
        IF EXISTS (SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = v_role AND rolcanlogin) THEN
            RAISE EXCEPTION 'agent boundary: existing role % can log in; use a fresh role name', v_role
                USING ERRCODE = 'invalid_parameter_value';
        END IF;
        IF EXISTS (
            SELECT 1
              FROM pg_catalog.pg_shdepend AS d
              JOIN pg_catalog.pg_roles AS r ON r.oid = d.refobjid
             WHERE d.refclassid = 'pg_catalog.pg_authid'::pg_catalog.regclass
               AND d.deptype = 'o'
               AND r.rolname = v_role
               AND NOT (d.dbid = (SELECT oid FROM pg_catalog.pg_database
                                   WHERE datname = pg_catalog.current_database())
                        AND ((d.classid = 'pg_catalog.pg_class'::pg_catalog.regclass
                              AND d.objid = pg_catalog.to_regclass('public.agent_pending_unknown'))
                          OR (d.classid = 'pg_catalog.pg_proc'::pg_catalog.regclass
                              AND d.objid IN (
                                  pg_catalog.to_regprocedure(
                                      'public.classify_unknown(integer, text, text, double precision, text)'),
                                  pg_catalog.to_regprocedure(
                                      'public.agent_is_pending_unknown(text, json)')))))
        ) THEN
            RAISE EXCEPTION 'agent boundary: existing role % owns objects besides the boundary; use a fresh role name', v_role
                USING ERRCODE = 'invalid_parameter_value';
        END IF;
    END LOOP;
END
$existing$;
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
-- Large objects are a write path that needs no table privilege, and PUBLIC
-- can execute these by default. (lo_import is superuser-only already.)
REVOKE EXECUTE ON FUNCTION
    pg_catalog.lo_create(oid),
    pg_catalog.lo_creat(integer),
    pg_catalog.lo_from_bytea(oid, bytea),
    pg_catalog.lo_import(text),
    pg_catalog.lo_import(text, oid),
    pg_catalog.lo_open(oid, integer),
    pg_catalog.lo_put(oid, bigint, bytea)
    FROM PUBLIC;

-- The owner role holds exactly what the view and classify_unknown need.
GRANT SELECT, UPDATE (metadata) ON TABLE public.survey_records TO {{owner_role}};

-- Start the boundary objects afresh: CREATE OR REPLACE would keep their
-- ACLs, and a view's column types cannot change in place. No CASCADE: an
-- object someone built on top of them makes the install fail loudly.
DROP VIEW IF EXISTS public.agent_pending_unknown;
-- Every overload, by name: a planted public.agent_is_pending_unknown(varchar,
-- json) would otherwise be the exact match for the varchar modality column.
DO $overloads$
DECLARE
    v_fn pg_catalog.regprocedure;
BEGIN
    FOR v_fn IN
        SELECT p.oid::pg_catalog.regprocedure
          FROM pg_catalog.pg_proc AS p
          JOIN pg_catalog.pg_namespace AS n ON n.oid = p.pronamespace
         WHERE n.nspname = 'public' AND p.proname IN ('agent_is_pending_unknown', 'classify_unknown')
    LOOP
        EXECUTE pg_catalog.format('DROP FUNCTION %s', v_fn);
    END LOOP;
END
$overloads$;

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
-- a hidden row. Every value is returned as text, uncast: one row holding a
-- value no cast could take (3e9 ms, a string sample rate) must not fail every
-- fetch. The agent parses and bound-checks them per record. Likewise a row
-- whose json holds a \u0000 escape, which makes every ->> on it fail
-- (22P05): it is left out, checked on the raw text first (CASE fixes the
-- order), and needs manual cleanup (agent/README.md). The plain modality
-- test keeps the modality index usable.
CREATE VIEW public.agent_pending_unknown
    WITH (security_barrier = true) AS
SELECT r.id,
       r.metadata ->> 'iq_snippet_path' AS iq_snippet_path,
       r.metadata ->> 'sample_rate' AS sample_rate,
       r.identifier ->> 'center_freq' AS center_freq,
       r.signal ->> 'peak_power' AS peak_power,
       r.metadata ->> 'snippet_duration_ms' AS snippet_duration_ms,
       r.metadata -> 'quality_flags' ->> 'snippet_rejected' AS snippet_rejected,
       r.metadata -> 'quality_flags' ->> 'snippet_dropped' AS snippet_dropped
  FROM public.survey_records AS r
 WHERE r.modality = 'unknown'
   AND CASE
           WHEN pg_catalog.strpos(r.metadata::text, E'\\u0000') > 0
             OR pg_catalog.strpos(r.identifier::text, E'\\u0000') > 0
             OR pg_catalog.strpos(r.signal::text, E'\\u0000') > 0 THEN false
           ELSE public.agent_is_pending_unknown(r.modality::text, r.metadata::json)
       END;
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
       AND CASE
               -- A \u0000 escape fails the jsonb cast below; such a row is
               -- not one the agent may write, as the view leaves it out.
               WHEN pg_catalog.strpos(r.metadata::text, E'\\u0000') > 0
                 OR pg_catalog.strpos(r.identifier::text, E'\\u0000') > 0
                 OR pg_catalog.strpos(r.signal::text, E'\\u0000') > 0 THEN false
               ELSE public.agent_is_pending_unknown(r.modality::text, r.metadata::json)
           END
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
