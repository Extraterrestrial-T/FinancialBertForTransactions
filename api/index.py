"""Vercel entry point for the PRAGMA-lite Flask research site.

Vercel discovers the top-level ``app`` object in this recognised function
location. The application itself remains in ``webapp.server`` so local Flask
and Vercel deployments run the same code.
"""

from webapp.server import app

