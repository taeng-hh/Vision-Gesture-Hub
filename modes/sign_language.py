"""Korean Sign Language model placeholder.

IMPORTANT: MediaPipe's canned gesture labels are NOT Korean Sign Language.
Do not produce fabricated translations or route signs to OS actions.
"""


class SignLanguageMode:
    def update(self, observation):
        n = len(observation.hands)
        return f"Hands tracked: {n} / KSL model not trained yet"
