def message_template(
    prompt: str, system_message: str = "You are a helpful AI assistant."
) -> str:
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\
        {system_message}<|eot_id|>\n\
        <|start_header_id|>user<|end_header_id|>\n\n\

        {prompt}<|eot_id|>\n\
        <|start_header_id|>assistant<|end_header_id|>\n"""
