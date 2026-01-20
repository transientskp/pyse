from sourcefinder.utility.sourceparams import (
    SourceParams,
    source_params_descriptions,
)


def test_sourceparams_descriptions_complete():
    source_parameters = {p.value for p in SourceParams}
    description_keys = source_params_descriptions.keys()

    missing = source_parameters - description_keys
    extra = description_keys - source_parameters

    # Ensure that all source parameters have a description.
    assert not missing, f"Missing descriptions for: {sorted(missing)}"

    # And assert that all descriptions correspond to a source parameter.
    assert not extra, f"Descriptions for unknown params: {sorted(extra)}"
