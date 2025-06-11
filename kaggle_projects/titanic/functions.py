
def test_output(message: RequestTestOutput) -> ResponseTestOutput:
    assert os.path.exists(message.output_filepath)    
    assert os.path.exists(message.reference_filepath)
    
    ref_df = pd.read_csv(message.reference_filepath)
    out_df = pd.read_csv(message.output_filepath)
    assert ref_df.shape == out_df.shape, f"Shape mismatch: {ref_df.shape} != {out_df.shape}"
    assert set(ref_df.columns) == set(out_df.columns), f"Columns mismatch: {set(ref_df.columns)} != {set(out_df.columns)}"
    assert out_df.dtypes.equals(ref_df.dtypes), f"Types mismatch: {out_df.dtypes} != {ref_df.dtypes}"
    
    return ResponseTestOutput(
        status="success",
        **message.model_dump()
    )