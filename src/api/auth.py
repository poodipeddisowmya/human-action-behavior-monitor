"""Optional shared-token access control.

Off unless ACCESS_TOKEN is set — localhost use stays exactly as before. It
exists for the cases where the dashboard is reachable from the internet: a
Kaggle notebook behind a public tunnel, or a cloud GPU server.

The token is accepted from, in order: `?token=` in the URL, the cookie that
URL sets, an `Authorization: Bearer` header, or `X-Access-Token`. The cookie
is what makes the scheme work in a browser at all: the <img> MJPEG stream and
the WebSocket cannot attach headers, but they do send cookies.
"""
from __future__ import annotations

import hmac
from html import escape

from fastapi import Request, WebSocket
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from config import settings

COOKIE = "ham_token"
# Liveness must answer without credentials: tunnels, Docker and load
# balancers probe it and have no token.
_OPEN_PATHS = {"/api/health"}


def _matches(candidate: str | None) -> bool:
    return bool(candidate) and hmac.compare_digest(candidate, settings.access_token)


def _presented(conn: Request | WebSocket) -> tuple[str | None, bool]:
    """(token, came_from_query_string)."""
    q = conn.query_params.get("token")
    if q:
        return q, True
    c = conn.cookies.get(COOKIE)
    if c:
        return c, False
    h = conn.headers.get("authorization", "")
    if h.lower().startswith("bearer "):
        return h[7:].strip(), False
    return conn.headers.get("x-access-token"), False


def websocket_allowed(ws: WebSocket) -> bool:
    if not settings.access_token:
        return True
    return _matches(_presented(ws)[0])


_LOGIN_PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Access required</title><style>
body{{margin:0;min-height:100vh;display:flex;align-items:center;justify-content:center;
background:#0b1220;color:#e2e8f0;font:15px system-ui,sans-serif}}
form{{background:#111a2e;border:1px solid #23314f;border-radius:12px;padding:28px;width:min(360px,90vw)}}
h1{{font-size:18px;margin:0 0 6px}}p{{color:#94a3b8;margin:0 0 18px}}
input{{width:100%;box-sizing:border-box;padding:10px;border-radius:8px;border:1px solid #23314f;
background:#0b1220;color:#e2e8f0;margin-bottom:12px}}
button{{width:100%;padding:10px;border:0;border-radius:8px;background:#3b82f6;color:#fff;font-weight:600}}
.e{{color:#f87171}}</style></head><body>
<form method="get" action="{path}"><h1>Human Action &amp; Behavior Monitor</h1>
<p class="{cls}">{msg}</p><input name="token" type="password" placeholder="Access token" autofocus>
<button>Open dashboard</button></form></body></html>"""


async def middleware(request: Request, call_next):
    if not settings.access_token or request.url.path in _OPEN_PATHS:
        return await call_next(request)

    token, from_query = _presented(request)
    if not _matches(token):
        if request.url.path.startswith(("/api", "/ws")):
            return JSONResponse({"detail": "access token required"}, status_code=401)
        msg, cls = (("That token is not valid.", "e") if token
                    else ("Enter the access token you were given.", ""))
        return HTMLResponse(_LOGIN_PAGE.format(path=escape(request.url.path), msg=msg, cls=cls),
                            status_code=401)

    if not from_query:
        return await call_next(request)

    if request.method == "GET" and not request.url.path.startswith(("/api", "/ws")):
        # A page opened with ?token=: set the cookie and redirect to the clean
        # URL, so the token is not left in the address bar of a shared screen.
        response = RedirectResponse(request.url.path or "/", status_code=303)
    else:
        response = await call_next(request)
    # Behind a TLS-terminating tunnel the app itself sees plain http;
    # the forwarded header says what the browser actually used.
    https = request.headers.get("x-forwarded-proto", request.url.scheme) == "https"
    response.set_cookie(COOKIE, token, httponly=True, samesite="lax",
                        secure=https, max_age=7 * 24 * 3600)
    return response
