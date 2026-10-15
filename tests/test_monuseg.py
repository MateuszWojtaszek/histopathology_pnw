import numpy as np

from histopathology_pnw.datasets.monuseg import polygons_to_instance_mask, read_polygons

XML = """<?xml version="1.0"?>
<Annotations><Annotation><Regions>
  <Region Id="1"><Vertices>
    <Vertex X="1" Y="1"/><Vertex X="5" Y="1"/><Vertex X="5" Y="5"/><Vertex X="1" Y="5"/>
  </Vertices></Region>
  <Region Id="2"><Vertices><Vertex X="10" Y="10"/><Vertex X="12" Y="12"/></Vertices></Region>
  <Region Id="3"><Vertices>
    <Vertex X="10" Y="10"/><Vertex X="15" Y="10"/><Vertex X="15" Y="15"/><Vertex X="10" Y="15"/>
  </Vertices></Region>
</Regions></Annotation></Annotations>
"""


def test_read_polygons(tmp_path):
    path = tmp_path / "a.xml"
    path.write_text(XML)
    polygons = read_polygons(path)
    assert [len(p) for p in polygons] == [4, 2, 4]
    assert polygons[0][1].tolist() == [5.0, 1.0]  # (x, y)


def test_mask_skips_degenerate_polygons_and_labels_consecutively(tmp_path):
    path = tmp_path / "a.xml"
    path.write_text(XML)
    mask, stats = polygons_to_instance_mask(read_polygons(path), (20, 20))
    assert mask.dtype == np.uint16
    assert set(np.unique(mask)) == {0, 1, 2}
    assert stats.n_too_few_vertices == 1
    assert stats.n_instances == 2
    assert mask[3, 3] == 1  # row = y, column = x
    assert mask[12, 12] == 2


def test_overlap_later_polygon_wins():
    square = np.array([[0, 0], [6, 0], [6, 6], [0, 6]], dtype=float)
    inner = np.array([[2, 2], [4, 2], [4, 4], [2, 4]], dtype=float)
    mask, stats = polygons_to_instance_mask([square, inner], (10, 10))
    assert mask[3, 3] == 2
    assert stats.n_overlap_pixels > 0
    assert stats.n_lost == 0


def test_fully_covered_instance_is_lost():
    small = np.array([[2, 2], [4, 2], [4, 4], [2, 4]], dtype=float)
    big = np.array([[0, 0], [6, 0], [6, 6], [0, 6]], dtype=float)
    mask, stats = polygons_to_instance_mask([small, big], (10, 10))
    assert stats.n_lost == 1
    assert mask.max() == 1


def test_vertices_outside_image_are_clipped():
    poly = np.array([[-5, -5], [5, -5], [5, 5], [-5, 5]], dtype=float)
    mask, stats = polygons_to_instance_mask([poly], (10, 10))
    assert stats.n_instances == 1
    assert mask[0, 0] == 1
