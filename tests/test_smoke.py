def test_packages_import():
    import capture.common  # noqa: F401
    import ingest  # noqa: F401
    import schema  # noqa: F401
    import storage  # noqa: F401
    import viz  # noqa: F401

    assert True
