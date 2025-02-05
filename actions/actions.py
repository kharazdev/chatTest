# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"

# from typing import Any, Text, Dict, List
#
# from rasa_sdk import Action, Tracker
# from rasa_sdk.executor import CollectingDispatcher
#
#
# class ActionHelloWorld(Action):
#
#     def name(self) -> Text:
#         return "action_hello_world"
#
#     def run(self, dispatcher: CollectingDispatcher,
#             tracker: Tracker,
#             domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
#
#         dispatcher.utter_message(text="Hello World!")
#
#         return []

# actions/custom_actions.py

from typing import Text, Dict, List, Any  # Add this import
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from transformers import AutoModelForCausalLM, AutoTokenizer

class ActionGenerateResponse(Action):
    def name(self) -> Text:
        return "action_generate_response"

    def __init__(self):
        # Load the pre-trained DialoGPT model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
        self.model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        # Get the user's latest message
        user_message = tracker.latest_message.get("text")

        # user_name = tracker.get_slot("user_name")
        # if user_name:
        #     dispatcher.utter_message(text=f"Hey {user_name}, what’s your favorite movie?")
        # else:
        #     dispatcher.utter_message(text="What’s your favorite movie?")   

        # Generate a response using DialoGPT
        inputs = self.tokenizer.encode(user_message, return_tensors="pt")
        outputs = self.model.generate(inputs, max_length=100, num_return_sequences=1)
        bot_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Send the response back to the user
        dispatcher.utter_message(text=bot_response)

        return []