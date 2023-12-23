import pyaudio
from google.cloud import speech
from google.oauth2 import service_account
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification


tokenizer = AutoTokenizer.from_pretrained("kredor/punctuate-all")
model = AutoModelForTokenClassification.from_pretrained("kredor/punctuate-all")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
credentials = service_account.Credentials.from_service_account_file('key.json')
client = speech.SpeechClient(credentials=credentials)
config = speech.RecognitionConfig(
    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
    sample_rate_hertz=44100,
    language_code='en-US',
    enable_automatic_punctuation=False,
)


def punctuate(text):
    # Tokenize the text
    tokenized_text = tokenizer.encode(text, add_special_tokens=False, return_tensors="pt")

    # Punctuate each sentence using the model
    punctuated_tokens = []
    for i in range(tokenized_text.shape[0]):
        # Get the tokens for the current sentence
        sentence_tokens = tokenized_text[i].tolist()

        # Remove padding tokens
        sentence_tokens = [t for t in sentence_tokens if t != tokenizer.pad_token_id]

        # Predict the punctuation marks
        outputs = model(torch.tensor([sentence_tokens])).logits.argmax(-1).tolist()

        # Insert the punctuation marks
        punctuated_sentence = ""
        for j, token in enumerate(sentence_tokens):
            if tokenizer.convert_ids_to_tokens([token])[0].startswith("##"):
                # Append to the previous word
                punctuated_sentence = punctuated_sentence[:-1] + tokenizer.convert_ids_to_tokens([token])[0][2:]
            else:
                # Append to the current word
                punctuated_sentence += tokenizer.convert_ids_to_tokens([token])[0]

            if j < len(outputs) and outputs[j] == 1:
                # Insert a punctuation mark if predicted score for adding it is greater than 0.8
                punctuated_sentence += "."

        punctuated_tokens.append(punctuated_sentence)

    # Combine the punctuated sentences
    punctuated_text = tokenizer.decode(tokenizer.encode(" ".join(punctuated_tokens)))
    punctuated_text = punctuated_text.replace(" .", ".")  # Remove spaces before periods

    return punctuated_text.replace("<s>", "").replace("</s>", "")




def stream_audio():
    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=44100,
        input=True,
        frames_per_buffer=1024
    )

    def audio_generator():
        print("Audio Start")
        while True:
            yield stream.read(1024)

    requests = (
        speech.StreamingRecognizeRequest(audio_content=content)
        for content in audio_generator()
    )

    streaming_config = speech.StreamingRecognitionConfig(
        config=config,
        interim_results=True,
    )

    responses = client.streaming_recognize(
        config=streaming_config,
        requests=requests,
    )

    for response in responses:
        for result in response.results:
            if not result.alternatives:
                continue
            transcript = result.alternatives[0].transcript
            if result.is_final:
                pass
                print(f" {punctuate(transcript)}")
            else:
                try:
                    if '.' in punctuate(transcript):
                        index = punctuate(transcript).index(".")
                        print(punctuate(transcript)[:index])
                except:
                    pass


    stream.stop_stream()
    stream.close()
    audio.terminate()


if __name__ == '__main__':
    stream_audio()