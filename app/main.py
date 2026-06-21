"""Streamlit delivery surface (placeholder).

The interactive application — impulse-response explorer, exposure maps, and the
estimand toggle — is built in a later phase. This placeholder lets the container
image and entry point exist from the scaffold onward.
"""

from __future__ import annotations


def main() -> None:
    """Render the placeholder application."""
    import streamlit as st

    st.title("Causal Impact Lab")
    st.write("The interactive application is under construction.")


if __name__ == "__main__":
    main()
