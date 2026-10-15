import pytest

from histopathology_pnw.models.student import build_student_init, sam_checkpoint_path


@pytest.mark.slow
@pytest.mark.skipif(not sam_checkpoint_path().is_file(), reason="SAM weights not pulled")
def test_student_init_loads_strictly(tmp_path):
    # build_student_init loads the SAM encoder with strict=True, so a key mismatch raises
    path = build_student_init(seed=0, out_path=tmp_path / "init.pth")
    assert path.stat().st_size > 1e9
