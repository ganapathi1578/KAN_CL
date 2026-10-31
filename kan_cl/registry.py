"""Small typed registry used to keep model and component choices configurable."""
from __future__ import annotations

from typing import Any, Callable, Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, name: str) -> None:
        self.name = name
        self._items: dict[str, T] = {}

    def register(self, name: str) -> Callable[[T], T]:
        def decorator(item: T) -> T:
            if name in self._items:
                raise KeyError(f"{name!r} is already registered in {self.name}")
            self._items[name] = item
            return item
        return decorator

    def get(self, name: str) -> T:
        try:
            return self._items[name]
        except KeyError as exc:
            choices = ", ".join(sorted(self._items))
            raise KeyError(f"Unknown {self.name} {name!r}. Available: {choices}") from exc

    def build(self, name: str, **kwargs: Any) -> Any:
        constructor = self.get(name)
        return constructor(**kwargs)  # type: ignore[operator]

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._items))

    def __contains__(self, name: str) -> bool:
        return name in self._items


MODELS: Registry[Any] = Registry("model")
BASIS_FUNCTIONS: Registry[Any] = Registry("basis function")
STRATEGIES: Registry[Any] = Registry("continual-learning strategy")
