from pyfe3d import BeamC, Spring

from kite_fem.FEMStructure import FEM_structure


def test_current_model_instantiates_only_beamc_and_spring_elements():
    structure = FEM_structure(
        initial_conditions=[
            [[0.0, 0.0, 0.0], [0, 0, 0], 1, True],
            [[1.0, 0.0, 0.0], [0, 0, 0], 1, False],
            [[2.0, 0.0, 0.0], [0, 0, 0], 1, False],
        ],
        spring_matrix=[[0, 1, 100.0, 0.0, 1.0, "default"]],
        pulley_matrix=[[0, 1, 2, 100.0, 0.0, 2.0]],
        beam_matrix=[[1, 2, 0.1, 0.5, 1.0]],
    )

    assert [type(element.spring) for element in structure.spring_elements] == [
        Spring,
        Spring,
        Spring,
    ]
    assert [element.springtype for element in structure.spring_elements] == [
        "default",
        "pulley",
        "pulley",
    ]
    assert [type(element.beam) for element in structure.beam_elements] == [BeamC]
