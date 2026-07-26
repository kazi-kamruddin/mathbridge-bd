APP_NAME = "MathBridge"
APP_CAPTION = (
    "A Gemma-powered handwritten STEM mistake detector, remediation tutor, "
    "and classroom misconception analyzer for Bangladesh."
)
MODEL_NAME = "gemma-4-26b-a4b-it"
MAX_TEACHER_ANSWERS = 20

LEVEL_GUIDANCE = {
    "Class 6-8": """
Use very simple Bangla and short sentences. Explain every symbol. Focus on one
basic idea at a time. Avoid formal proofs. Give one easy correction example and
one very easy retry problem.
""",
    "SSC": """
Use clear Bangladesh textbook-style Bangla. Name the relevant chapter or rule.
Show the first wrong step, the corrected step, and concise exam-oriented advice.
Give one SSC-style retry problem.
""",
    "HSC": """
Use precise academic Bangla. Explain the underlying rule, intermediate reasoning,
and relevant restrictions or assumptions. Give one conceptual and one
calculation-oriented retry task.
""",
    "University": """
Use technical Bangla with useful English terminology in parentheses. State formal
assumptions, identify invalid transformations, discuss alternative interpretations,
and give an analytical extension problem.
""",
}

STATUS_OPTIONS = [
    "Fully Correct",
    "Partially Correct",
    "Fully Incorrect",
    "Incomplete or Unclear",
]

ERROR_CATEGORIES = [
    "No Error",
    "Arithmetic Error",
    "Sign Error",
    "Distribution or Bracket Error",
    "Fraction Operation Error",
    "Exponent or Root Error",
    "Missing Term",
    "Wrong Formula or Method",
    "Invalid Algebraic Transformation",
    "Equation Balancing Error",
    "Unit or Notation Error",
    "Conceptual Error",
    "Incomplete Solution",
    "Handwriting Unclear",
    "Other",
]
