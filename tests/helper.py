import pytest
import functools


def skip_if_import_exception(function):
    """Assist in skipping tests failing because of missing dependencies."""
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except ImportError as err:
            pytest.skip(str(err))
    return wrapper


def start_http_server(redirect=None):
    """Returns the port of the newly started HTTP server."""
    def _do_redirect(handler, location):
        handler.send_response(301)
        handler.send_header("Location", location)
        handler.end_headers()

    import socket
    import sys
    import threading
    # Start HTTP server to serve TAR files.
    # pylint:disable=g-import-not-at-top
    if sys.version_info[0] == 2:
        import BaseHTTPServer
        import SimpleHTTPServer

        class HTTPServerV6(BaseHTTPServer.HTTPServer):

            address_family = socket.AF_INET6

        class RedirectHandler(SimpleHTTPServer.SimpleHTTPRequestHandler):

            def do_GET(self):
                _do_redirect(self, redirect)

        server = HTTPServerV6(("", 0), RedirectHandler if redirect else
                              SimpleHTTPServer.SimpleHTTPRequestHandler)
        server_port = server.server_port
    else:
        import http.server
        import socketserver

        class TCPServerV6(socketserver.TCPServer):

            address_family = socket.AF_INET6

        class RedirectHandler(http.server.SimpleHTTPRequestHandler):

            def do_GET(self):
                _do_redirect(self, redirect)

        server = TCPServerV6(("", 0), RedirectHandler if redirect else
                             http.server.SimpleHTTPRequestHandler)
        _, server_port, _, _ = server.server_address
    # pylint:disable=g-import-not-at-top

    thread = threading.Thread(target=server.serve_forever)
    thread.daemon = True
    thread.start()

    return server_port
