import inspect
import functools
from typing import Callable, Type, Any, Dict
from pydantic import BaseModel, Field, create_model
from pydantic.fields import FieldInfo
from pydantic import ValidationError


class Tool:
    def __init__(
        self, name: str, description: str, parameters: Type[BaseModel], func: Callable
    ):
        functools.update_wrapper(self, func)

        self.name = name
        self.description = description
        self.parameters = parameters
        self.func = func

    def __call__(self, *args, **kwargs):
        """Allow Tool instance to be called directly like a function: grep()"""
        # Get function signature
        sig = inspect.signature(self.func)

        # Use bind_partial to allow partial parameter binding
        bound_args = sig.bind_partial(*args, **kwargs)
        bound_args.apply_defaults()

        # Correct parameter values: for FieldInfo, the default value should be FieldInfo.default, not the FieldInfo object itself
        corrected_args = {}
        for param_name, value in bound_args.arguments.items():
            # If the parameter value is a FieldInfo object, it indicates an issue during default value extraction
            if isinstance(value, FieldInfo):
                corrected_args[param_name] = value.default
            else:
                corrected_args[param_name] = value

        # Validate parameters using Pydantic model to ensure proper handling of Field defaults
        try:
            validated_params = self.parameters.model_validate(corrected_args)
            return self.func(**validated_params.model_dump())
        except ValidationError as e:
            # Return a user-friendly error message instead of raising the exception
            error_details = []
            for error in e.errors():
                field = error["loc"][-1] if error["loc"] else "unknown"
                msg = error["msg"]
                input_val = error.get("input", "unknown")
                error_details.append(
                    f"Field '{field}': {msg} (received: {repr(input_val)})"
                )

            error_message = (
                f"Validation failed for tool '{self.name}': {'; '.join(error_details)}"
            )
            return {"error": error_message, "success": False}

    def execute(self, **kwargs) -> Any:
        """Entry point for Agent calls: with Pydantic validation"""
        try:
            validated_params = self.parameters.model_validate(kwargs)
            return self.func(**validated_params.model_dump())
        except ValidationError as e:
            # Return a user-friendly error message instead of raising the exception
            error_details = []
            for error in e.errors():
                field = error["loc"][-1] if error["loc"] else "unknown"
                msg = error["msg"]
                input_val = error.get("input", "unknown")
                error_details.append(
                    f"Field '{field}': {msg} (received: {repr(input_val)})"
                )

            error_message = (
                f"Validation failed for tool '{self.name}': {'; '.join(error_details)}"
            )
            return {"error": error_message, "success": False}

    def to_openai_format(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters.model_json_schema(),
            },
        }


def tool(func: Callable) -> Tool:
    name = func.__name__
    description = inspect.getdoc(func) or "No description provided."

    sig = inspect.signature(func)
    fields = {}

    for param_name, param in sig.parameters.items():
        annotation = (
            param.annotation if param.annotation != inspect.Parameter.empty else Any
        )
        default = param.default

        if isinstance(default, FieldInfo):
            # When the default value is FieldInfo, use it directly
            fields[param_name] = (annotation, default)
        elif default == inspect.Parameter.empty:
            # When there is no default value, mark as required
            fields[param_name] = (annotation, Field(...))
        else:
            # When there is a regular default value, create a new Field
            fields[param_name] = (annotation, Field(default))

    parameters_model = create_model(f"{name}Params", **fields)

    return Tool(
        name=name, description=description, parameters=parameters_model, func=func
    )
