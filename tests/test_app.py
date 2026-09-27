"""
Integration tests for Streamlit interface and search behavior.

Authors: Anish Velagapudi, Izad Khokhar, Aurick Smart
"""

from pathlib import Path
from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def test_streamlit_app_starts_and_search_suggestions_update_as_typed() -> None:
    app = AppTest.from_file(str(APP_PATH), default_timeout=60).run()

    assert not app.exception
    assert not app.error
    assert any("STEER" in title.value for title in app.title)
    assert len(app.text_input) >= 1
    assert len(app.pills) == 1
    assert len(app.dataframe) >= 2

    # Verify search input filter
    app.text_input[0].set_value("D H Conley").run()
    assert not app.exception
    assert not app.error
    assert len(app.pills[0].options) == 1

    app.text_input[0].set_value("High School").run()
    assert not app.exception
    assert len(app.pills[0].options) == 5

    app.text_input[0].set_value("no matching school name").run()
    assert not app.exception
    assert not app.error
    assert any("No matching institutions" in message.value for message in app.warning)