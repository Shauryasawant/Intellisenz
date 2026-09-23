import os
import unittest

import data


class TestInfluxSettings(unittest.TestCase):
    def test_load_influx_settings_from_environment(self):
        original = {
            "INFLUX_URL": os.environ.get("INFLUX_URL"),
            "INFLUX_TOKEN": os.environ.get("INFLUX_TOKEN"),
            "INFLUX_ORG": os.environ.get("INFLUX_ORG"),
            "INFLUX_BUCKET": os.environ.get("INFLUX_BUCKET"),
        }
        try:
            os.environ["INFLUX_URL"] = "http://localhost:8086"
            os.environ["INFLUX_TOKEN"] = "test-token"
            os.environ["INFLUX_ORG"] = "test-org"
            os.environ["INFLUX_BUCKET"] = "test-bucket"

            settings = data.load_influx_settings()
            self.assertEqual(settings.url, "http://localhost:8086")
            self.assertEqual(settings.token, "test-token")
            self.assertEqual(settings.org, "test-org")
            self.assertEqual(settings.bucket, "test-bucket")
        finally:
            for key, value in original.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


if __name__ == "__main__":
    unittest.main()
