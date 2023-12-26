Solution
========

This is how I approached the solution to the problem.

## Rationale

+ I started researching the [`microsoft/phi-2`](https://huggingface.co/microsoft/phi-2) model from huggingface and I started doing some testing to check how it performed.
  + I learned about the different configurations the model can be used for
  + I discovered the model is trained mainly for English usage, and it's not reliable for other languages. Despite having experienced some skill with Spanish, this moves me towards implementing an English-only chat.
+ I searched for a quick way of implementing a chatbot-like UI with gradion, and I find [this](https://www.gradio.app/guides/creating-a-chatbot-fast#example-using-a-local-open-source-llm-with-hugging-face)
+ I try and test the [chat template](https://www.gradio.app/guides/creating-a-chatbot-fast#example-using-a-local-open-source-llm-with-hugging-face), and start making some changes so that it works with `phi-2`.
  + First, I change the messages format so that it fits phi-2 format.

    ```
    Alice: I don't know why, I'm struggling to maintain focus while studying. Any suggestions?
    Bob: Well, have you tried creating a study schedule and sticking to it?
    Alice: Yes, I have, but it doesn't seem to help much.
    Bob: Hmm, maybe you should try studying in a quiet environment, like the library.
    Alice: ...
    ```

  + In addition, I change the names of user and bot. From `"\n<human>:"+item[0], "\n<bot>:"+item[1]`, to `"\nUser:"+item[0], "\nBot:"+item[1]`
  + It's better to strip the variable `messages`, as I'm not sure breaklines are not harmful.
  + The `StoppingCriteria` class needs to be updated so that it accounts for the differences between `microsoft/phi-2` and `togethercomputer/RedPajama-INCITE-Chat-3B-v1`.
    + Change the stopping tokens. Get a list of specal tokens with `tokenizer.all_special_tokens`, `tokenizer.all_special_ids`, and `tokenizer.vocab_size`. I discover phi has it's special tokens at the end of the vocab.
    + Also, I realise phi-2 tends to write a lot and continues the conversation inventing turns. Thus, I make the breakline `\n` as a stopping character as well.
