#!/usr/bin/env python
#
# Copyright 2016 - The Android Open Source Project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for acloud.internal.lib.utils."""

import getpass
import os
import subprocess

import mock

import unittest
from acloud.internal.lib import driver_test_lib
from acloud.internal.lib import utils


class UtilsTest(driver_test_lib.BaseDriverTest):

  def testCreateSshKeyPair_KeyAlreadyExists(self):
    """Test when the key pair already exists."""
    public_key = "/fake/public_key"
    private_key = "/fake/private_key"
    self.Patch(os.path, "exists", side_effect=lambda path: path == public_key)
    self.Patch(subprocess, "check_call")
    utils.CreateSshKeyPairIfNotExist(private_key, public_key)
    self.assertEqual(subprocess.check_call.call_count, 0)

  def testCreateSshKeyPair_KeyAreCreated(self):
    """Test when the key pair created."""
    public_key = "/fake/public_key"
    private_key = "/fake/private_key"
    self.Patch(os.path, "exists", return_value=False)
    self.Patch(subprocess, "check_call")
    self.Patch(os, "rename")
    utils.CreateSshKeyPairIfNotExist(private_key, public_key)
    self.assertEqual(subprocess.check_call.call_count, 1)
    subprocess.check_call.assert_called_with(
        utils.SSH_KEYGEN_CMD + ["-C", getpass.getuser(), "-f", private_key],
        stdout=mock.ANY, stderr=mock.ANY)


if __name__ == "__main__":
    unittest.main()
