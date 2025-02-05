# # This files contains your custom actions which can be used to run
# # custom Python code.
# #
# # See this guide on how to implement these action:
# # https://rasa.com/docs/rasa/custom-actions


# # This is a simple example for a custom action which utters "Hello World!"

# # from typing import Any, Text, Dict, List
# #
# # from rasa_sdk import Action, Tracker
# # from rasa_sdk.executor import CollectingDispatcher
# #
# #
# # class ActionHelloWorld(Action):
# #
# #     def name(self) -> Text:
# #         return "action_hello_world"
# #
# #     def run(self, dispatcher: CollectingDispatcher,
# #             tracker: Tracker,
# #             domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
# #
# #         dispatcher.utter_message(text="Hello World!")
# #
# #         return []

# # actions/custom_actions.py

# from typing import Text, Dict, List, Any  # Add this import
# from rasa_sdk import Action, Tracker
# from rasa_sdk.executor import CollectingDispatcher
# from transformers import AutoModelForCausalLM, AutoTokenizer

# class ActionGenerateResponse(Action):
#     def name(self) -> Text:
#         return "action_generate_response"

#     def __init__(self):
#         # Load the pre-trained DialoGPT model and tokenizer
#         self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
#         self.model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")

#     def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
#         # Get the user's latest message
#         user_message = tracker.latest_message.get("text")

#         # user_name = tracker.get_slot("user_name")
#         # if user_name:
#         #     dispatcher.utter_message(text=f"Hey {user_name}, what’s your favorite movie?")
#         # else:
#         #     dispatcher.utter_message(text="What’s your favorite movie?")   

#         # Generate a response using DialoGPT
#         inputs = self.tokenizer.encode(user_message, return_tensors="pt")
#         outputs = self.model.generate(inputs, max_length=100, num_return_sequences=1)
#         bot_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

#         # Send the response back to the user
#         dispatcher.utter_message(text=bot_response)

#         return []


from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from transformers import AutoModelForCausalLM, AutoTokenizer

class ActionLightChatResponse(Action):
    def __init__(self):
        # Initialize DialoGPT-small
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
        self.model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")

    def name(self) -> Text:
        return "action_generate_response"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any]
    ) -> List[Dict[Text, Any]]:
        # Get the latest user message
        user_message = tracker.latest_message.get("text")

        # Encode the input
        inputs = self.tokenizer.encode(user_message + self.tokenizer.eos_token, return_tensors="pt")
        
        # Generate response
        response_ids = self.model.generate(
            inputs,
            max_length=1000,
            pad_token_id=self.tokenizer.eos_token_id,
            no_repeat_ngram_size=3,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7
        )

        # Decode the response
        response = self.tokenizer.decode(response_ids[0], skip_special_tokens=True)

        # Send the response
        dispatcher.utter_message(text=response)

        return []