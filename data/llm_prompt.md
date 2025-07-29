## Token-level Task
```
{
      "role": "user",
      "content": f'''You are a Dockerfile code prediction assistant.

      Given the following Dockerfile content and its file path context, your task is to complete the next Dockerfile tokens. 
      
      For each token, return the top 10 most likely candidates.

      Please respond with an array of arrays. Each inner array contains 10 different tokens (strings) representing the top-10 predictions for that position.

      The format should be exactly:
      [
        ["token1", "token2", ..., "token10"],
        ["token1", "token2", ..., "token10"],
        ...
      ]

      Do not include any explanation or extra text.

      Tokens in each inner array must be different.

      Do not respond empty array.

      Input:
      {item['input_ids']}
      '''
}
```

## Line-level Task
```
{
      "role": "user",
      "content": f'''Given the following Dockerfile content and file path context, predict the next 
      line of Dockerfile content. Only return one token sequence without any explanation.
      Input: {inputs}
      '''
}
```
