import os
import requests


PROMPT_TEMPLATES = {
    "spear_phishing": (
        "Write a spear phishing email targeting a {role} at {company}. "
        "Use a {tone} tone. Impersonate {impersonate}. "
        "Include a fake urgent request to click a link and verify credentials. "
        "Keep it under 200 words. Return only the email body."
    ),
    "generic_phishing": (
        "Write a phishing email pretending to be from {impersonate}. "
        "Use urgency: the recipient's account will be suspended. "
        "Include a suspicious link. Tone: {tone}. Under 150 words. Return only the email body."
    ),
    "whaling": (
        "Write a whaling phishing email targeting a C-suite executive (CEO/CFO). "
        "Impersonate {impersonate}. Request an urgent wire transfer or credential reset. "
        "Tone: {tone}. Under 200 words. Return only the email body."
    ),
}

#this class generates synthetic phishing email variants using an LLM API

class AttackSimulator:
   
    def __init__(self, llm_provider: str = "openai"):
        self.prompt_template: str = ""
        self.llm_provider: str = llm_provider   # "openai" | "claude"
        self.generated: list[str] = []

    #  Public API                                                          
    def build_prompt(
        self,
        attack_type: str = "generic_phishing",
        tone: str = "urgent and professional",
        impersonate: str = "PayPal Security Team",
        role: str = "employee",
        company: str = "Acme Corp",
    ) -> str:
        template = PROMPT_TEMPLATES.get(attack_type, PROMPT_TEMPLATES["generic_phishing"])
        self.prompt_template = template.format(
            tone=tone, impersonate=impersonate, role=role, company=company
        )
        return self.prompt_template

    #Sends prompt to the configured LLM and returns the generated text
    def call_llm_api(self, prompt: str) -> str:
        if self.llm_provider == "claude":
            return self._call_claude(prompt)
        return self._call_openai(prompt)

    def get_variants(
        self,
        attack_type: str = "generic_phishing",
        n: int = 3,
        **kwargs,
    ) -> list[str]:
        #Generates n synthetic phishing email variants
        self.generated = []
        tones = ["urgent and threatening", "friendly and professional", "formal and authoritative"]
        for i in range(n):
            kwargs["tone"] = tones[i % len(tones)]
            prompt = self.build_prompt(attack_type=attack_type, **kwargs)
            try:
                email = self.call_llm_api(prompt)
                self.generated.append(email)
            except Exception as e:
                self.generated.append(f"[Generation failed: {e}]")
        return self.generated

    #  Private helpers                                                     
    def _call_openai(self, prompt: str) -> str:
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY not set.")
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 300,
            "temperature": 0.9,
        }
        resp = requests.post("https://api.openai.com/v1/chat/completions", json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()

    def _call_claude(self, prompt: str) -> str:
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise EnvironmentError("ANTHROPIC_API_KEY not set.")
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        payload = {
            "model": "claude-3-haiku-20240307",
            "max_tokens": 300,
            "messages": [{"role": "user", "content": prompt}],
        }
        resp = requests.post("https://api.anthropic.com/v1/messages", json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        return resp.json()["content"][0]["text"].strip()