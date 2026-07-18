# tests/test_graphic_interaction.py
# MIT License - Copyright (c) 2025-2026 Luc Prevost
# Tests for the interactive overlay repositioning (2D, keyboard-driven).

import pytest

from topoptcomec.gui.main_window import MainWindow


@pytest.fixture(autouse=True)
def _prevent_loss_csv_writes():
    """Keep GUI unit tests from overwriting checked-in result histories."""
    from unittest.mock import patch

    with patch(
        "topoptcomec.gui.main_window.exporters.save_loss", return_value=(True, None)
    ):
        yield


def _make_window(qt_app) -> MainWindow:
    """Build a MainWindow on a 2D preset and force a replot."""
    window = MainWindow()
    # Force a 2D problem (default preset is already 2D, but make it explicit).
    window.dim_widget.nz.setValue(0)
    window.replot()
    return window


def test_actor_per_element_map_populated(qt_app):
    """Each active force/support produces its own actor entry in the map."""
    window = _make_window(qt_app)
    # The default preset (ForceInverter_2Sup_2D) has 1 input + 1 output force
    # and 2 supports, all active.
    pf = window.last_params["Forces"]
    n_active_in = sum(1 for d in pf["fidir"] if d != "-")
    n_active_out = sum(1 for d in pf["fodir"] if d != "-")
    n_active_sup = sum(1 for d in window.last_params["Supports"]["sdim"] if d != "-")
    expected = n_active_in + n_active_out + n_active_sup
    assert len(window._overlay_actor_map) == expected
    window.close()


def test_can_interact_with_overlays_2d_default(qt_app):
    """Default 2D problem with no worker running: interaction allowed."""
    window = _make_window(qt_app)
    assert window._can_interact_with_overlays() is True
    assert window._can_move_selected() is True
    window.close()


def test_can_move_selected_blocked_in_3d(qt_app):
    """3D plot blocks XY-plane movement but still allows selection."""
    window = _make_window(qt_app)
    window.dim_widget.nz.setValue(4)
    # Trigger a parameter refresh so last_params reflects the 3D mode.
    window.on_parameter_changed()
    assert window._can_move_selected() is False
    # The base gate (picking / rotation) remains open in 3D.
    assert window._can_interact_with_overlays() is True
    window.close()


def test_can_interact_with_overlays_blocked_during_deformation(qt_app):
    """Deformation view disables the interactive overlay repositioning."""
    window = _make_window(qt_app)
    window.is_displaying_deformation = True
    assert window._can_interact_with_overlays() is False
    window.close()


def test_can_interact_with_overlays_blocked_when_worker_running(qt_app):
    """A running worker (any non-None) disables the interaction."""
    window = _make_window(qt_app)

    class _FakeWorker:
        def wait(self):
            pass

    window.worker = _FakeWorker()
    try:
        assert window._can_interact_with_overlays() is False
    finally:
        window.worker = None
    window.close()


def test_on_overlay_picked_sets_selection(qt_app):
    """Picking a known overlay actor records (kind, idx) in _selected_overlay."""
    window = _make_window(qt_app)
    # Find the first actor from the map and feed it back to the callback.
    actor_id = next(iter(window._overlay_actor_map))
    expected = window._overlay_actor_map[actor_id]
    # The map stores id(actor); the callback receives the actor itself. We
    # must pass the original actor instance, so look it up in the live list.
    actor = next(a for a in window._overlay_actors if id(a) == actor_id)
    window._on_overlay_picked(actor)
    assert window._selected_overlay == expected
    assert window._highlight_actor is not None
    window.close()


def test_on_overlay_picked_ignores_unknown_actor(qt_app):
    """An actor not in the overlay map does not change the selection."""
    window = _make_window(qt_app)
    window._selected_overlay = None

    class _BogusActor:
        pass

    window._on_overlay_picked(_BogusActor())
    assert window._selected_overlay is None
    window.close()


def test_deselect_overlay_clears_state(qt_app):
    """_deselect_overlay resets the logical selection and drops the highlight."""
    window = _make_window(qt_app)
    actor_id = next(iter(window._overlay_actor_map))
    actor = next(a for a in window._overlay_actors if id(a) == actor_id)
    window._on_overlay_picked(actor)
    assert window._selected_overlay is not None
    assert window._highlight_actor is not None
    window._deselect_overlay()
    assert window._selected_overlay is None
    assert window._highlight_actor is None
    window.close()


def test_move_selected_updates_spinbox_values(qt_app):
    """Moving right by 1 increments the underlying spinbox by 1."""
    window = _make_window(qt_app)
    # Select the first input force.
    actor_id = next(
        i for i, v in window._overlay_actor_map.items() if v[0] == "force_in"
    )
    actor = next(a for a in window._overlay_actors if id(a) == actor_id)
    window._on_overlay_picked(actor)
    _, idx = window._selected_overlay
    x_w = window.forces_widget.input_forces[idx]["fix"]
    y_w = window.forces_widget.input_forces[idx]["fiy"]
    old_x, old_y = x_w.value(), y_w.value()
    # Move right (+1 in x). Disable shift so step is 1.
    window._move_selected(1, 0)
    assert x_w.value() == old_x + 1
    assert y_w.value() == old_y
    window.close()


def test_move_selected_clamps_to_domain(qt_app):
    """Moving past the right edge clamps to nelx instead of overshooting."""
    window = _make_window(qt_app)
    nelx = window.last_params["Dimensions"]["nelxyz"][0]
    # Pick the first input force and shove it to the right edge.
    actor_id = next(
        i for i, v in window._overlay_actor_map.items() if v[0] == "force_in"
    )
    actor = next(a for a in window._overlay_actors if id(a) == actor_id)
    window._on_overlay_picked(actor)
    _, idx = window._selected_overlay
    x_w = window.forces_widget.input_forces[idx]["fix"]
    x_w.setValue(nelx)
    window.on_parameter_changed()  # rebuild actors + map at the new position
    # The selection was dropped by replot, re-pick the same element.
    actor_id = next(
        i for i, v in window._overlay_actor_map.items() if v[0] == "force_in"
    )
    actor = next(a for a in window._overlay_actors if id(a) == actor_id)
    window._on_overlay_picked(actor)
    window._move_selected(1, 0)  # would overshoot by 1
    assert x_w.value() == nelx
    window.close()


def test_move_selected_no_op_when_worker_running(qt_app):
    """Keyboard move is a no-op while a worker is running."""
    window = _make_window(qt_app)
    actor_id = next(
        i for i, v in window._overlay_actor_map.items() if v[0] == "force_in"
    )
    actor = next(a for a in window._overlay_actors if id(a) == actor_id)
    window._on_overlay_picked(actor)
    _, idx = window._selected_overlay
    x_w = window.forces_widget.input_forces[idx]["fix"]
    old_x = x_w.value()

    class _FakeWorker:
        def wait(self):
            pass

    window.worker = _FakeWorker()
    try:
        window._move_selected(1, 0)
        assert x_w.value() == old_x  # unchanged
    finally:
        window.worker = None
    window.close()


def test_move_selected_no_op_without_selection(qt_app):
    """Calling _move_selected with no selection is a safe no-op."""
    window = _make_window(qt_app)
    window._selected_overlay = None
    window._move_selected(1, 0)  # must not raise
    window.close()


def test_overlay_position_widgets_for_each_kind(qt_app):
    """_overlay_position_widgets returns the right spinboxes for each kind."""
    window = _make_window(qt_app)
    # force_in: returns (x, y, z) widgets
    widgets = window._overlay_position_widgets("force_in", 0)
    assert widgets is not None
    assert len(widgets) == 3
    assert widgets[0] is window.forces_widget.input_forces[0]["fix"]
    assert widgets[1] is window.forces_widget.input_forces[0]["fiy"]
    assert widgets[2] is window.forces_widget.input_forces[0]["fiz"]
    # force_out
    widgets = window._overlay_position_widgets("force_out", 0)
    assert widgets is not None
    assert widgets[0] is window.forces_widget.output_forces[0]["fox"]
    # support
    widgets = window._overlay_position_widgets("support", 0)
    assert widgets is not None
    assert widgets[0] is window.supports_widget.inputs[0]["sx"]
    # out of range
    assert window._overlay_position_widgets("force_in", 999) is None
    # unknown kind
    assert window._overlay_position_widgets("bogus", 0) is None
    window.close()


def test_selection_survives_replot(qt_app):
    """Re-applying highlight after a replot keeps _selected_overlay set."""
    window = _make_window(qt_app)
    actor_id = next(iter(window._overlay_actor_map))
    actor = next(a for a in window._overlay_actors if id(a) == actor_id)
    window._on_overlay_picked(actor)
    selection = window._selected_overlay
    assert selection is not None
    # Replot drops all overlay actors and rebuilds the map; the logical
    # selection must survive and a new highlight actor must be created.
    window.replot()
    assert window._selected_overlay == selection
    assert window._highlight_actor is not None
    window.close()


def test_setup_interactive_overlay_tools_is_idempotent(qt_app):
    """Calling _setup_interactive_overlay_tools twice does not double-register."""
    window = _make_window(qt_app)
    # PyVista stores key callbacks on the RenderWindowInteractor, not the
    # plotter itself.
    n_before = len(window.plotter.iren._key_press_event_callbacks.get("Left", []))
    window._setup_interactive_overlay_tools()
    n_after = len(window.plotter.iren._key_press_event_callbacks.get("Left", []))
    assert n_after == n_before  # no new registration on second call
    window.close()


def test_arrow_keys_clear_default_zoom_bindings(qt_app):
    """PyVista's default zoom_camera on Up/Down must be cleared by our setup."""
    window = _make_window(qt_app)
    # Each arrow key should have exactly one callback (our move handler).
    # If PyVista's zoom_camera were still bound, the count would be > 1.
    for key in ("Left", "Right", "Up", "Down"):
        callbacks = window.plotter.iren._key_press_event_callbacks.get(key, [])
        assert len(callbacks) == 1, (
            f"{key} should have exactly our move callback, got {len(callbacks)}"
        )
    window.close()


def _select_first_force(window, kind: str = "force_in"):
    """Helper: pick the first actor of the given kind and return (kind, idx)."""
    actor_id = next(i for i, v in window._overlay_actor_map.items() if v[0] == kind)
    actor = next(a for a in window._overlay_actors if id(a) == actor_id)
    window._on_overlay_picked(actor)
    return window._selected_overlay


def test_rotate_selected_force_cycles_in_2d(qt_app):
    """In 2D, 'r' rotates X:→ → Y:↓ → X:← → Y:↑ → X:→ (clockwise)."""
    window = _make_window(qt_app)
    _select_first_force(window, "force_in")
    combo = window.forces_widget.input_forces[0]["fidir"]
    # Default preset has fidir = "Y:↑" (index 3). One 'r' should go to X:→.
    # Set to known start (X:→, index 1).
    combo.setCurrentIndex(1)
    window.on_parameter_changed()
    # re-pick since replot dropped the selection
    _select_first_force(window, "force_in")
    # Clockwise cycle [1, 4, 2, 3]: X:→ (1) → Y:↓ (4)
    window._rotate_selected_force()
    assert combo.currentIndex() == 4  # Y:↓
    window._rotate_selected_force()
    assert combo.currentIndex() == 2  # X:←
    window._rotate_selected_force()
    assert combo.currentIndex() == 3  # Y:↑
    window._rotate_selected_force()
    assert combo.currentIndex() == 1  # back to X:→
    window.close()


def test_rotate_selected_force_cycles_in_3d(qt_app):
    """In 3D, 'r' advances through all 6 directions sequentially."""
    window = _make_window(qt_app)
    window.dim_widget.nz.setValue(4)
    window.on_parameter_changed()  # refresh last_params with nelz=4
    _select_first_force(window, "force_in")
    combo = window.forces_widget.input_forces[0]["fidir"]
    # 3D cycle is [1, 2, 3, 4, 5, 6].
    combo.setCurrentIndex(1)
    window.on_parameter_changed()
    _select_first_force(window, "force_in")
    window._rotate_selected_force()
    assert combo.currentIndex() == 2  # X:←
    window._rotate_selected_force()
    assert combo.currentIndex() == 3  # Y:↑
    window._rotate_selected_force()
    assert combo.currentIndex() == 4  # Y:↓
    window._rotate_selected_force()
    assert combo.currentIndex() == 5  # Z:<
    window._rotate_selected_force()
    assert combo.currentIndex() == 6  # Z:>
    window._rotate_selected_force()
    assert combo.currentIndex() == 1  # wrap to X:→
    window.close()


def test_rotate_selected_force_no_op_on_support(qt_app):
    """Pressing 'r' with a support selected does nothing (supports don't rotate)."""
    window = _make_window(qt_app)
    _select_first_force(window, "support")
    # _can_rotate_selected_force requires a force selection.
    assert window._can_rotate_selected_force() is False
    # Call should be a safe no-op (no crash, no change).
    for sw in window.supports_widget.inputs:
        old_idx = sw["sdim"].currentIndex()
        window._rotate_selected_force()
        assert sw["sdim"].currentIndex() == old_idx
        break
    window.close()


def test_rotate_selected_force_no_op_without_selection(qt_app):
    """Pressing 'r' with no selection is a safe no-op."""
    window = _make_window(qt_app)
    window._selected_overlay = None
    combo = window.forces_widget.input_forces[0]["fidir"]
    old_idx = combo.currentIndex()
    window._rotate_selected_force()
    assert combo.currentIndex() == old_idx
    window.close()


def test_rotate_selected_force_no_op_when_worker_running(qt_app):
    """Rotation is blocked while a worker is running."""
    window = _make_window(qt_app)
    _select_first_force(window, "force_in")
    combo = window.forces_widget.input_forces[0]["fidir"]
    combo.setCurrentIndex(1)
    window.on_parameter_changed()
    _select_first_force(window, "force_in")

    class _FakeWorker:
        def wait(self):
            pass

    window.worker = _FakeWorker()
    try:
        window._rotate_selected_force()
        assert combo.currentIndex() == 1  # unchanged
    finally:
        window.worker = None
    window.close()


def test_rotate_selected_force_works_in_3d_selection_persists(qt_app):
    """In 3D, picking a force keeps it selected so 'r' can rotate it."""
    window = _make_window(qt_app)
    window.dim_widget.nz.setValue(4)
    window.on_parameter_changed()
    # Selection should be possible in 3D (base gate allows it).
    assert window._can_interact_with_overlays() is True
    _select_first_force(window, "force_in")
    assert window._selected_overlay is not None
    # Movement is blocked in 3D but rotation is allowed.
    assert window._can_move_selected() is False
    assert window._can_rotate_selected_force() is True
    window.close()


def test_rotate_activates_inactive_force(qt_app):
    """'r' on an inactive ("-") force activates it at the cycle's first entry."""
    window = _make_window(qt_app)
    # Pick the first force then set its direction to "-".
    _select_first_force(window, "force_in")
    combo = window.forces_widget.input_forces[0]["fidir"]
    combo.setCurrentIndex(0)  # "-"
    window.on_parameter_changed()
    # The inactive force has no actor, so selection is gone. Force a stale
    # logical selection that points at the same widget.
    window._selected_overlay = ("force_in", 0)
    window._rotate_selected_force()
    # First cycle element in 2D is index 1 (X:→).
    assert combo.currentIndex() == 1
    window.close()
