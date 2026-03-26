from datetime import datetime

from . import tool


@tool
def get_current_date() -> str:
    """
    Get the current date and time

    Returns:
        A string describing the current date and time in format: YYYY-MM-DD HH:MM:SS
    """
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")


if __name__ == "__main__":
    print(get_current_date.to_openai_format())
    print(get_current_date())
