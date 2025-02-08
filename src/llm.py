import openai
from loguru import logger
from time import sleep
from api_keys import OPENAI_API_KEY, DEEPGRAM_API_KEY

from constants import OUTPUT_FILE_NAME, SYSTEM_PROMPT, SHORTER_INSTRACT, LONGER_INSTRACT
from deepgram import (
    DeepgramClient,
    PrerecordedOptions,
    FileSource,
    LiveTranscriptionEvents,
    LiveOptions,
    Microphone,
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


is_finals = []
def live_listen_and_transcribe(WINDOW, listenTime: int):
    try:
        deepgram: DeepgramClient = DeepgramClient(DEEPGRAM_API_KEY)
        dg_connection = deepgram.listen.websocket.v("1")

        def on_open(self, open, **kwargs):
            print("Connection Open")

        def on_message(self, result, **kwargs):
            print("IN on message")
            global is_finals
            sentence = result.channel.alternatives[0].transcript
            if len(sentence) == 0:
                return
            if result.is_final:
                print(f"Message: {result.to_json()}")
                is_finals.append(sentence)

                # Speech Final means we have detected sufficient silence to consider this end of speech
                # Speech final is the lowest latency result as it triggers as soon an the endpointing value has triggered
                if result.speech_final:
                    utterance = " ".join(is_finals)
                    print(f"Speech Final: {utterance}")
                    WINDOW.write_event_value("-WHISPER COMPLETED-", utterance)
                    is_finals = []
                else:
                    # These are useful if you need real time captioning and update what the Interim Results produced
                    print(f"Is Final: {sentence}")
            else:
                # These are useful if you need real time captioning of what is being spoken
                print(f"Interim Results: {sentence}")

        def on_metadata(self, metadata, **kwargs):
            print(f"Metadata: {metadata}")

        def on_speech_started(self, speech_started, **kwargs):
            print("Speech Started")

        def on_utterance_end(self, utterance_end, **kwargs):
            print("Utterance End")
            global is_finals
            if len(is_finals) > 0:
                utterance = " ".join(is_finals)
                print(f"Utterance End: {utterance}")
                is_finals = []

        def on_close(self, close, **kwargs):
            print("Connection Closed")

        def on_error(self, error, **kwargs):
            print(f"Handled Error: {error}")

        def on_unhandled(self, unhandled, **kwargs):
            print(f"Unhandled Websocket Message: {unhandled}")

        dg_connection.on(LiveTranscriptionEvents.Open, on_open)
        dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)
        dg_connection.on(LiveTranscriptionEvents.Metadata, on_metadata)
        dg_connection.on(LiveTranscriptionEvents.SpeechStarted, on_speech_started)
        dg_connection.on(LiveTranscriptionEvents.UtteranceEnd, on_utterance_end)
        dg_connection.on(LiveTranscriptionEvents.Close, on_close)
        dg_connection.on(LiveTranscriptionEvents.Error, on_error)
        dg_connection.on(LiveTranscriptionEvents.Unhandled, on_unhandled)

        options: LiveOptions = LiveOptions(
            model="nova-2",
            language="en-US",
            # Apply smart formatting to the output
            smart_format=True,
            # Raw audio format details
            encoding="linear16",
            channels=1,
            sample_rate=16000,
            # To get UtteranceEnd, the following must be set:
            interim_results=True,
            utterance_end_ms="1000",
            vad_events=True,
            # Time in milliseconds of silence to wait for before finalizing speech
            endpointing=300,
        )

        addons = {
            # Prevent waiting for additional numbers
            "no_delay": "true"
        }

        print("\n\nPress Enter to stop recording...\n\n")
        if dg_connection.start(options, addons=addons) is False:
            print("Failed to connect to Deepgram")
            return

        microphone = Microphone(dg_connection.send)
        microphone.start()

        # I'd like to continue this until input is given, but python threads crash if you take keyboard input in 2 places. 
        # Just listen for the passed number of seconds
        sleep(listenTime)

        microphone.finish()
        dg_connection.finish()

        print("Finished")
        sleep(10)
        print("Really done!")

    except Exception as e:
        print(f"Could not open socket: {e}")
        return

