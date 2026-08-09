from ingest.grid import grid_cell_key


def test_nearby_points_share_a_grid_cell():
    key_a = grid_cell_key(47.60620, -122.33210)
    key_b = grid_cell_key(47.60623, -122.33208)
    assert key_a == key_b


def test_distant_points_have_different_grid_cells():
    key_a = grid_cell_key(47.6062, -122.3321)
    key_b = grid_cell_key(48.0000, -121.0000)
    assert key_a != key_b
