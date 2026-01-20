from sourcefinder.utility.sourceparams import (
    SourceParams,
    source_params_descriptions,
)


def test_sourceparams_descriptions_complete():
    # Ensure that all source parameters have a description.
    assert all(p.value in source_params_descriptions for p in SourceParams)
    # And assert that all descriptions correspond to a source parameter.
    assert all(p in SourceParams for p in source_params_descriptions)
