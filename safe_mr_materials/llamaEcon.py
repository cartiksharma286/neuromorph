from groq import Groq

client = Groq()

chat_completion = client.chat.completions.create(
    messages=[
        # Set an optional system message. This sets the behavior of the
        # assistant and can be used to provide specific instructions for
        # how it should behave throughout the conversation.
        {
            "role": "system",
            "content": "You are a climate change economist."
        },
        # Set a user message for the assistant to respond to.
        {
            "role": "user",
            "content": "Generate optimal cost economics for the 2030 Canadian climate change initiative"
        }
    ],

    # The language model which will generate the completion.
#    model="llama-3.3-70b-versatile"
     model="llama-3.1-8b-instant"
#    model="openai/gpt-oss-120b"
)

# Print the completion returned by the LLM.
print(chat_completion.choices[0].message.content)
