import openai
from loguru import logger
from api_keys import OPENAI_API_KEY, DEEPGRAM_API_KEY

from constants import OUTPUT_FILE_NAME, SYSTEM_PROMPT, SHORTER_INSTRACT, LONGER_INSTRACT
from deepgram import (
    DeepgramClient,
    PrerecordedOptions,
    FileSource,
)


openai.api_key = OPENAI_API_KEY

def transcribe_audio(path_to_file: str = OUTPUT_FILE_NAME) -> str:
    """
    Transcribes an audio file into text.

    Args:
        path_to_file (str, optional): The path to the audio file to be transcribed.

    Returns:
        str: The transcribed text.

    Raises:
        Exception: If the audio file fails to transcribe.
    """
    with open(path_to_file, "rb") as audio_file:
        try:
            # STEP 1 Create a Deepgram client using the API key
            deepgram = DeepgramClient( DEEPGRAM_API_KEY)

            payload: FileSource = {
                "buffer": audio_file.read(),
            }

            #STEP 2: Configure Deepgram options for audio analysis
            options = PrerecordedOptions(
                model="nova-3",
                smart_format=True,
            )

            # STEP 3: Call the transcribe_file method with the text payload and options
            response = deepgram.listen.rest.v("1").transcribe_file(payload, options)

            # STEP 4: Print the response
            print(response.to_json(indent=4))
            
        except Exception as error:
            logger.error(f"Can't transcribe audio: {error}")
            raise error
    return response["results"]["channels"][0]['alternatives'][0]['transcript']


def generate_answer(transcript: str, short_answer: bool = True, temperature: float = 0.7) -> str:
    """
    Generates an answer based on the given transcript using the OpenAI GPT-3.5-turbo model.

    Args:
        transcript (str): The transcript to generate an answer from.
        short_answer (bool): Whether to generate a short answer or not. Defaults to True.
        temperature (float): The temperature parameter for controlling the randomness of the generated answer.

    Returns:
        str: The generated answer.

    Example:
        ```python
        transcript = "Can you tell me about the weather?"
        answer = generate_answer(transcript, short_answer=False, temperature=0.8)
        print(answer)
        ```

    Raises:
        Exception: If the LLM fails to generate an answer.
    """
    if short_answer:
        system_prompt = SYSTEM_PROMPT + SHORTER_INSTRACT
    else:
        system_prompt = SYSTEM_PROMPT + LONGER_INSTRACT
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": transcript},
            ],
        )
    except Exception as error:
        logger.error(f"Can't generate answer: {error}")
        raise error
    return response["choices"][0]["message"]["content"]
