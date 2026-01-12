from typing import List, Dict

class ChatFormatter:
    """
    Handles the conversion of conversation history into a structured prompt 
    that Plagon LLM can understand.
    
    Format Style (XML-like, robust for instruction tuning):
    
    <system>
    {system_message}
    </system>
    
    <user>
    {user_message}
    </user>
    
    <assistant>
    {assistant_message}
    </assistant>
    """
    
    SYSTEM_START = "<system>\n"
    SYSTEM_END = "\n</system>\n"
    USER_START = "<user>\n"
    USER_END = "\n</user>\n"
    ASSISTANT_START = "<assistant>\n"
    ASSISTANT_END = "\n</assistant>\n"
    
    @staticmethod
    def format_turn(role: str, content: str) -> str:
        content = content.strip()
        if role == "system":
            return f"{ChatFormatter.SYSTEM_START}{content}{ChatFormatter.SYSTEM_END}"
        elif role == "user":
            return f"{ChatFormatter.USER_START}{content}{ChatFormatter.USER_END}"
        elif role == "assistant":
            return f"{ChatFormatter.ASSISTANT_START}{content}{ChatFormatter.ASSISTANT_END}"
        else:
            raise ValueError(f"Unknown role: {role}")

    @classmethod
    def format_history(cls, history: List[Dict[str, str]]) -> str:
        """
        Compiles the entire history into a single string.
        Should be called before tokenization.
        """
        prompt = ""
        for message in history:
            prompt += cls.format_turn(message['role'], message['content'])
        
        # Prepare for the assistant's next response
        prompt += cls.ASSISTANT_START.strip() + "\n"
        return prompt

    @staticmethod
    def get_stop_sequences() -> List[str]:
        """
        Returns strings that should signal the model to STOP generating.
        Crucial so the model doesn't hallucinate the User's next turn.
        """
        return ["</assistant>", "<user>", "<system>"]
