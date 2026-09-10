import json
import os
import re
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from core.schema import CleaningAction
import yaml

load_dotenv()

SYSTEM_PROMPT = """You are a concise data cleaning assistant.
Analyze the provided data issues (in JSON) and recommend cleaning steps.
You MUST reply with a JSON array of objects.

Each object in your JSON array must have:
- "column": column name (or "__dataset__" for duplicate rows)
- "action": short action name (e.g., "impute median", "cap outliers", "drop duplicates", "standardize text")
- "why": 1 short sentence explanation

ANSWER ONLY IN JSON FORMAT
Keep it under 75 words total."""


class RecommendationModule:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f).get("recommendation-llm", {})
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            self.llm = None
        else:
            self.llm = ChatGoogleGenerativeAI(
                model=self.config.get("model", "gemini-2.5-flash"),
                temperature=self.config.get("temperature", 0.1),
                google_api_key=api_key
            )

    def recommend(self, issues: list) -> tuple[str, list[CleaningAction], bool]:
        if not self.llm:
            return "Error: GOOGLE_API_KEY is missing from .env file.", [], False

        # 1. Format issues into simple JSON for the LLM
        issues_json = json.dumps([
            {
                "column": i.column,
                "issue": i.issue_type,
                "affected_pct": f"{i.affected_pct:.1%}",
                "description": i.description
            } for i in issues
        ])

        # 2. Call Gemini
        try:
            prompt = f"{SYSTEM_PROMPT}\n\nData Issues:\n{issues_json}"
            response = self.llm.invoke(prompt)
            raw_text = response.content
            
            # Extract JSON from response
            json_text = self._extract_json_text(raw_text)
            recommendations = json.loads(json_text)
            
            # 3. Map JSON/keywords to our system's CleaningAction objects
            actions = self._map_to_actions(recommendations)
            
            # Format chat summary
            chat_text = f"**LLM Recommended Plan:**\n"
            for item in recommendations:
                chat_text += f"- **{item.get('column')}**: {item.get('action')} — *{item.get('why')}*\n"

            return chat_text, actions, True

        except Exception as e:
            return f"Error connecting to LLM: {str(e)}", [], False

    def _extract_json_text(self, text: str) -> str:
        """Finds JSON array inside code blocks or raw text."""
        match = re.search(r"\[[\s\S]*\]", text)
        if match:
            return match.group(0)
        return text

    def _map_to_actions(self, recommendations: list[dict]) -> list[CleaningAction]:
        mapped_actions = []

        for item in recommendations:
            col = item.get("column", "")
            act_text = str(item.get("action", "")).lower()
            why = item.get("why", "")

            # Keyword matching to system operations
            if "duplicate" in act_text:
                mapped_actions.append(CleaningAction(
                    column="__dataset__", operation="remove_duplicates", 
                    params={"keep": "first"}, source="llm", rationale=why
                ))
            elif "impute" in act_text or "fill" in act_text or "missing" in act_text:
                strategy = "mode" if "mode" in act_text else "median"
                mapped_actions.append(CleaningAction(
                    column=col, operation="impute_missing", 
                    params={"strategy": strategy}, source="llm", rationale=why
                ))
            elif "outlier" in act_text or "cap" in act_text:
                mapped_actions.append(CleaningAction(
                    column=col, operation="cap_outliers", 
                    params={"method": "iqr", "multiplier": 1.5}, source="llm", rationale=why
                ))
            elif "drop" in act_text or "delete" in act_text:
                mapped_actions.append(CleaningAction(
                    column=col, operation="drop_column", 
                    params={}, source="llm", rationale=why
                ))
            elif "standardize" in act_text or "text" in act_text or "string" in act_text:
                mapped_actions.append(CleaningAction(
                    column=col, operation="standardize_strings", 
                    params={}, source="llm", rationale=why
                ))
            elif "cast" in act_text or "type" in act_text or "numeric" in act_text:
                mapped_actions.append(CleaningAction(
                    column=col, operation="cast_type", 
                    params={"target_type": "float"}, source="llm", rationale=why
                ))

        return mapped_actions