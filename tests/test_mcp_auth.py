import unittest
import json
from io import StringIO
from unittest.mock import patch
from saguaro.mcp.server import MCPServer

class TestMCPAuth(unittest.TestCase):
    def test_no_auth_required(self):
        server = MCPServer()
        self.assertTrue(server.authenticated)
        
    def test_auth_required_valid(self):
        server = MCPServer(auth_token="secret123")
        self.assertFalse(server.authenticated)
        
        # Simulate initialize request with token
        init_req = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "authorization": "secret123"
            }
        }
        
        # Initialize
        _ = server.handle_initialize(init_req["params"])
        self.assertTrue(server.authenticated)
        
    def test_auth_required_invalid(self):
        server = MCPServer(auth_token="secret123")
        
        # Simulate initialize request with WRONG token
        init_req = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "authorization": "wrong"
            }
        }
        
        with self.assertRaises(ValueError):
             server.handle_initialize(init_req["params"])
             
    def test_unauthenticated_request(self):
        server = MCPServer(auth_token="secret123")
        
        # Mock stdout
        with patch('sys.stdout', new=StringIO()) as fake_out:
            # Send a tool call without auth (and NOT initialize)
            req = {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/list",
                "params": {}
            }
            # We want to call handle_message, which checks auth
            # but handle_message reads line string.
            server.handle_message(json.dumps(req))
            
            output = fake_out.getvalue()
            response = json.loads(output)
            
            self.assertIn("error", response)
            self.assertEqual(response["error"]["code"], -32002) # Unauthenticated

if __name__ == "__main__":
    unittest.main()
