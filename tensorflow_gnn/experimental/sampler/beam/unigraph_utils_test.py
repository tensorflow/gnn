# Copyright 2023 The TensorFlow GNN Authors. All Rights Reserved.
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
# ==============================================================================
"""Tests for unigraph reading utils."""

import apache_beam as beam
from apache_beam.testing import test_pipeline
from apache_beam.testing import util
import numpy as np
import tensorflow as tf
import tensorflow_gnn as tfgnn
from tensorflow_gnn.data import unigraph
from tensorflow_gnn.experimental.sampler.beam import unigraph_utils
from tensorflow_gnn.utils import test_utils

from google.protobuf import text_format


_CUSTOMER_IDS = b"""
  1876448 1372437 1368305 1974494 1257724 1758057 1531660 1489311 1407706
  196838 1195675 1659366 1499004 1344333 1443888 1108778 175583 1251872
  1493851 1599418 1768701 1549489 1879799 125454
""".split()


class UnigraphUtilsTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self.resource_dir = test_utils.get_resource_dir('testdata/heterogeneous')
    self.seed_path = test_utils.get_resource(
        'testdata/heterogeneous/customer.csv')
    self.graph_schema_file = unigraph.find_schema_filename(self.resource_dir)

  def test_read_seeds(self):
    data_path = self.seed_path
    expected_seeds = [
        (
            bytes(f'S{id}', 'utf-8'),
            [np.array([id], dtype=np.object_), np.array([1], dtype=np.int64)],
        )
        for id in _CUSTOMER_IDS
    ]
    with test_pipeline.TestPipeline() as root:
      seeds = unigraph_utils.read_seeds(root, data_path)
      util.assert_that(
          seeds | beam.Keys(),
          util.equal_to([seed[0] for seed in expected_seeds]),
      )
      root.run()

  def test_convert_unigraph_edge_features(self):
    # Read the schema.
    schema = tfgnn.read_schema(self.graph_schema_file)
    with self.assertRaisesRegex(
        NotImplementedError, 'Edge features are not currently supported'
    ):
      with test_pipeline.TestPipeline() as root:
        _ = root | unigraph_utils.ReadAndConvertUnigraph(
            schema, self.resource_dir
        )
        root.run()

  def test_convert_unigraph_to_sampler_v2(self):
    schema = text_format.Parse(
        """
      node_sets {
        key: "customer"
        value {
          metadata {
            filename: "customer.csv"
          }
          features {
            key: "name"
            value: {
              description: "Name"
              dtype: DT_STRING
            }
          }
          features {
            key: "address"
            value: {
              description: "address"
              dtype: DT_STRING
            }
          }
          features {
            key: "zipcode"
            value: {
              description: "Zipcode"
              dtype: DT_INT64
            }
          }
          features {
            key: "score"
            value: {
              description: "Credit score"
              dtype: DT_FLOAT
            }
          }
        }
      }
      node_sets {
        key: "creditcard"
        value {
          metadata {
            filename: "creditcard.csv"
          }
          features {
            key: "number"
            value: {
              description: "Credit card number"
              dtype: DT_INT64
            }
          }
          features {
            key: "issuer"
            value: {
              description: "Credit card issuer institution"
              dtype: DT_STRING
            }
          }
        }
      }
      edge_sets {
        key: "owns_card"
        value {
          description: "Owns and uses the credit card."
          source: "customer"
          target: "creditcard"
          metadata {
            filename: "owns_card.csv"
          }
        }
      }""",
        tfgnn.GraphSchema(),
    )
    with test_pipeline.TestPipeline() as root:
      expected_dict = {
          'nodes/customer': [
              (
                  b'1876448',
                  (
                      b'\n\x86\x01\n\x19\n\x04name\x12\x11\n\x0f\n\rJi'
                      b' Grindstaff\n\x12\n\x03#id\x12\x0b\n\t\n\x071876448\n\x11\n\x05score\x12\x08\x12\x06\n\x04\xb8z\x04?\n\x12\n\x07zipcode\x12\x07\x1a\x05\n\x03\xd8\xd9\x04\n.\n\x07address\x12#\n!\n\x1f343'
                      b' Third St. Houston, TX 77016'
                  ),
              ),
              (
                  b'1372437',
                  (
                      b'\n\x8c\x01\n\x1a\n\x04name\x12\x12\n\x10\n\x0eAugustina'
                      b" Uren\n4\n\x07address\x12)\n'\n%9940 Prairie Ave. Deer"
                      b' Park, NY'
                      b' 11729\n\x11\n\x05score\x12\x08\x12\x06\n\x04\x91\xd1g?\n\x11\n\x07zipcode\x12\x06\x1a\x04\n\x02\xd1[\n\x12\n\x03#id\x12\x0b\n\t\n\x071372437'
                  ),
              ),
              (
                  b'1368305',
                  (
                      b'\n\x89\x01\n\x12\n\x03#id\x12\x0b\n\t\n\x071368305\n\x11\n\x05score\x12\x08\x12\x06\n\x04o\x92v>\n\x11\n\x07zipcode\x12\x06\x1a\x04\n\x02\x99\\\n3\n\x07address\x12(\n&\n$478'
                      b' Grove Drive Hicksville, NY'
                      b' 11801\n\x18\n\x04name\x12\x10\n\x0e\n\x0cYolonda Nave'
                  ),
              ),
              (
                  b'1974494',
                  (
                      b'\n\x8b\x01\n\x11\n\x05score\x12\x08\x12\x06\n\x04\xd5\xbf\xaf>\n\x12\n\x03#id\x12\x0b\n\t\n\x071974494\n\x12\n\x07zipcode\x12\x07\x1a\x05\n\x03\x96\xf8\x02\n\x1c\n\x04name\x12\x14\n\x12\n\x10Adriana'
                      b' Mcburney\n0\n\x07address\x12%\n#\n!909 Vermont St.'
                      b' Livonia, MI 48150'
                  ),
              ),
              (
                  b'1257724',
                  (
                      b'\n\x8b\x01\n\x16\n\x04name\x12\x0e\n\x0c\n\nDione'
                      b" Reeb\n\x12\n\x03#id\x12\x0b\n\t\n\x071257724\n6\n\x07address\x12+\n)\n'7758"
                      b' West Devon St. Algonquin, IL'
                      b' 60102\n\x11\n\x05score\x12\x08\x12\x06\n\x04\x08"\n?\n\x12\n\x07zipcode\x12\x07\x1a\x05\n\x03\xc6\xd5\x03'
                  ),
              ),
              (
                  b'1758057',
                  (
                      b'\n\x83\x01\n\x12\n\x03#id\x12\x0b\n\t\n\x071758057\n\x11\n\x05score\x12\x08\x12\x06\n\x04\x07Um?\n\x12\n\x07zipcode\x12\x07\x1a\x05\n\x03\x84\xeb\x01\n.\n\x07address\x12#\n!\n\x1f720'
                      b' Marsh Road Tucker, GA'
                      b' 30084\n\x16\n\x04name\x12\x0e\n\x0c\n\nGeri Bones'
                  ),
              ),
              (
                  b'1531660',
                  (
                      b'\n\x87\x01\n\x11\n\x05score\x12\x08\x12\x06\n\x04\xa2\xa3?>\n/\n\x07address\x12$\n"\n'
                      b' 227 Bishop St. Bemidji, MN'
                      b' 56601\n\x19\n\x04name\x12\x11\n\x0f\n\rKrystal'
                      b' Pablo\n\x12\n\x07zipcode\x12\x07\x1a\x05\n\x03\x99\xba\x03\n\x12\n\x03#id\x12\x0b\n\t\n\x071531660'
                  ),
              ),
              (
                  b'1489311',
                  (
                      b'\n\x88\x01\n\x18\n\x04name\x12\x10\n\x0e\n\x0cTonia'
                      b' Behnke\n\x11\n\x05score\x12\x08\x12\x06\n\x04\xe2\xf3\xef=\n\x12\n\x07zipcode\x12\x07\x1a\x05\n\x03\x86\xea\x02\n1\n\x07address\x12&\n$\n"51'
                      b' Ramblewood St. Hobart, IN'
                      b' 46342\n\x12\n\x03#id\x12\x0b\n\t\n\x071489311'
                  ),
              ),
              (
                  b'1407706',
                  (
                      b"\n\x89\x01\n\x12\n\x07zipcode\x12\x07\x1a\x05\n\x03\xd8\xea\x01\n\x11\n\x05score\x12\x08\x12\x06\n\x04\xd5\x93q>\n2\n\x07address\x12'\n%\n#714"
                      b' Hilldale Ave. Cumming, GA'
                      b' 30040\n\x12\n\x03#id\x12\x0b\n\t\n\x071407706\n\x18\n\x04name\x12\x10\n\x0e\n\x0cFidel'
                      b' Speers'
                  ),
              ),
              (
                  b'196838',
                  (
                      b'\n\x8f\x01\n\x1a\n\x04name\x12\x12\n\x10\n\x0eNecole'
                      b' Hunkins\n\x11\n\x05score\x12\x08\x12\x06\n\x04uMw=\n7\n\x07address\x12,\n*\n(14'
                      b' South Grove St. Coatesville, PA'
                      b' 19320\n\x11\n\x03#id\x12\n\n\x08\n\x06196838\n\x12\n\x07zipcode\x12\x07\x1a\x05\n\x03\xf8\x96\x01'
                  ),
              ),
              (
                  b'1195675',
                  (
                      b'\n\x86\x01\n\x12\n\x03#id\x12\x0b\n\t\n\x071195675\n\x11\n\x05score\x12\x08\x12\x06\n\x04c\xb8/?\n\x12\n\x07zipcode\x12\x07\x1a\x05\n\x03\xc5\x94\x01\n\x16\n\x04name\x12\x0e\n\x0c\n\nTona'
                      b' Crays\n1\n\x07address\x12&\n$\n"136 Somerset Dr.'
                      b' Chester, PA 19013'
                  ),
              ),
              (
                  b'1659366',
                  (
                      b'\n\x86\x01\n\x16\n\x04name\x12\x0e\n\x0c\n\nMary'
                      b' Zeitz\n\x12\n\x03#id\x12\x0b\n\t\n\x071659366\n\x11\n\x05score\x12\x08\x12\x06\n\x04}\xb8$?\n\x12\n\x07zipcode\x12\x07\x1a\x05\n\x03\xe4\xf6\x05\n1\n\x07address\x12&\n$\n"6'
                      b' Thatcher St. Hillsboro, OR 97124'
                  ),
              ),
              (
                  b'1499004',
                  (
                      b'\n\x91\x01\n\x11\n\x05score\x12\x08\x12\x06\n\x04q\x9dj?\n\x12\n\x03#id\x12\x0b\n\t\n\x071499004\n3\n\x07address\x12(\n&\n$673'
                      b' Elmwood Drive Fairburn, GA'
                      b' 30213\n\x1f\n\x04name\x12\x17\n\x15\n\x13Rudolph'
                      b' Sinquefield\n\x12\n\x07zipcode\x12\x07\x1a\x05\n\x03\x85\xec\x01'
                  ),
              ),
              (
                  b'1344333',
                  (
                      b'\n\x89\x01\n/\n\x07address\x12$\n"\n 7284 Young Lane'
                      b' Upland, CA'
                      b' 91784\n\x12\n\x03#id\x12\x0b\n\t\n\x071344333\n\x1b\n\x04name\x12\x13\n\x11\n\x0fMarhta'
                      b' Rodrigue\n\x11\n\x05score\x12\x08\x12\x06\n\x04s\xcd"?\n\x12\n\x07zipcode\x12\x07\x1a\x05\n\x03\x88\xcd\x05'
                  ),
              ),
              (
                  b'1443888',
                  (
                      b"\n\x8d\x01\n\x11\n\x05score\x12\x08\x12\x06\n\x04q\xea\xee>\n\x12\n\x03#id\x12\x0b\n\t\n\x071443888\n2\n\x07address\x12'\n%\n#19"
                      b' Rock Creek St. Sulphur, LA'
                      b' 70663\n\x1c\n\x04name\x12\x14\n\x12\n\x10Ricardo'
                      b' Bundrick\n\x12\n\x07zipcode\x12\x07\x1a\x05\n\x03\x87\xa8\x04'
                  ),
              ),
              (
                  b'1108778',
                  (
                      b'\n\x88\x01\n\x19\n\x04name\x12\x11\n\x0f\n\rMyron'
                      b' Barrick\n0\n\x07address\x12%\n#\n!450 Boston Street'
                      b' Solon, OH'
                      b' 44139\n\x12\n\x03#id\x12\x0b\n\t\n\x071108778\n\x11\n\x05score\x12\x08\x12\x06\n\x04\x90^7?\n\x12\n\x07zipcode\x12\x07\x1a\x05\n\x03\xeb\xd8\x02'
                  ),
              ),
              (
                  b'175583',
                  (
                      b'\n\x87\x01\n\x1a\n\x04name\x12\x12\n\x10\n\x0eNichol'
                      b' Poulton\n\x11\n\x03#id\x12\n\n\x08\n\x06175583\n\x11\n\x07zipcode\x12\x06\x1a\x04\n\x02\x90R\n0\n\x07address\x12%\n#\n!34'
                      b' Old 8th Drive Carmel, NY'
                      b' 10512\n\x11\n\x05score\x12\x08\x12\x06\n\x04\xdd\xa0\x95>'
                  ),
              ),
              (
                  b'1251872',
                  (
                      b'\n\x8e\x01\n\x12\n\x03#id\x12\x0b\n\t\n\x071251872\n\x12\n\x07zipcode\x12\x07\x1a\x05\n\x03\xf5\xac\x03\n7\n\x07address\x12,\n*\n(7351'
                      b' North Spring Ave. Oshkosh, WI'
                      b' 54901\n\x11\n\x05score\x12\x08\x12\x06\n\x04\xf9\xcb\xf7>\n\x18\n\x04name\x12\x10\n\x0e\n\x0cBoyd'
                      b' Padilla'
                  ),
              ),
              (
                  b'1493851',
                  (
                      b'\n\x8e\x01\n\x11\n\x05score\x12\x08\x12\x06\n\x04\x0bVC?\n\x12\n\x07zipcode\x12\x07\x1a\x05\n\x03\xa8\xd4\x02\n5\n\x07address\x12*\n(\n&7805'
                      b' Newport Street Sylvania, OH'
                      b' 43560\n\x1a\n\x04name\x12\x12\n\x10\n\x0eOrpha'
                      b' Yokoyama\n\x12\n\x03#id\x12\x0b\n\t\n\x071493851'
                  ),
              ),
              (
                  b'1599418',
                  (
                      b'\n\x8b\x01\n\x19\n\x04name\x12\x11\n\x0f\n\rUlysses'
                      b" Harps\n\x12\n\x03#id\x12\x0b\n\t\n\x071599418\n4\n\x07address\x12)\n'\n%9138"
                      b' Gates Street Braintree, MA'
                      b' 02184\n\x11\n\x05score\x12\x08\x12\x06\n\x04\x08\xc4\xb2>\n\x11\n\x07zipcode\x12\x06\x1a\x04\n\x02\x84\n'
                  ),
              ),
              (
                  b'1768701',
                  (
                      b'\n\x8d\x01\n\x12\n\x07zipcode\x12\x07\x1a\x05\n\x03\xfc\x81\x02\n5\n\x07address\x12*\n(\n&42'
                      b' South Wayne St. Hollywood, FL'
                      b' 33020\n\x11\n\x05score\x12\x08\x12\x06\n\x04\x10\xb76?\n\x12\n\x03#id\x12\x0b\n\t\n\x071768701\n\x19\n\x04name\x12\x11\n\x0f\n\rSulema'
                      b' Aguero'
                  ),
              ),
              (
                  b'1549489',
                  (
                      b'\n\x8f\x01\n\x11\n\x05score\x12\x08\x12\x06\n\x04\x96z\x01?\n5\n\x07address\x12*\n(\n&98'
                      b' Corona Court Morton Grove, IL'
                      b' 60053\n\x12\n\x07zipcode\x12\x07\x1a\x05\n\x03\x95\xd5\x03\n\x12\n\x03#id\x12\x0b\n\t\n\x071549489\n\x1b\n\x04name\x12\x13\n\x11\n\x0fMeredith'
                      b' Warman'
                  ),
              ),
              (
                  b'1879799',
                  (
                      b'\n\x83\x01\n\x12\n\x07zipcode\x12\x07\x1a\x05\n\x03\x83\xd8\x02\n\x17\n\x04name\x12\x0f\n\r\n\x0bVonda'
                      b' Borth\n\x11\n\x05score\x12\x08\x12\x06\n\x04\xda\x9d(?\n-\n\x07address\x12"\n'
                      b' \n\x1e97 Tunnel Dr. Elyria, OH'
                      b' 44035\n\x12\n\x03#id\x12\x0b\n\t\n\x071879799'
                  ),
              ),
              (
                  b'125454',
                  (
                      b'\n\x96\x01\n\x1a\n\x04name\x12\x12\n\x10\n\x0eCandida'
                      b' Uvalle\n\x12\n\x07zipcode\x12\x07\x1a\x05\n\x03\xad\xf7\x02\n\x11\n\x05score\x12\x08\x12\x06\n\x04\x07m+?\n\x11\n\x03#id\x12\n\n\x08\n\x06125454\n>\n\x07address\x123\n1\n/608'
                      b' Heritage Street Harrison Township, MI 48045'
                  ),
              ),
          ],
          'nodes/creditcard': [
              (
                  b'11238474857489380',
                  b'\nK\n\x1c\n\x03#id\x12\x15\n\x13\n\x1111238474857489380\n\x16\n\x06number\x12\x0c\x1a\n\n\x08\xe4\xc7\xca\xad\xd5\xaa\xfb\x13\n\x13\n\x06issuer\x12\t\n\x07\n\x05BofBC',
              ),
              (
                  b'14216252633958570',
                  b'\nM\n\x1c\n\x03#id\x12\x15\n\x13\n\x1114216252633958570\n\x16\n\x06number\x12\x0c\x1a\n\n\x08\xaa\x91\xa3\x82\xb4\xb3\xa0\x19\n\x15\n\x06issuer\x12\x0b\n\t\n\x07HeyBank',
              ),
              (
                  b'14541017563963440',
                  b'\nP\n\x1c\n\x03#id\x12\x15\n\x13\n\x1114541017563963440\n\x16\n\x06number\x12\x0c\x1a\n\n\x08\xb0\xc0\xca\xd4\xa7\x9f\xea\x19\n\x18\n\x06issuer\x12\x0e\n\x0c\n\nBellsGarbo',
              ),
              (
                  b'13549061668422190',
                  b'\nP\n\x18\n\x06issuer\x12\x0e\n\x0c\n\nBellsGarbo\n\x1c\n\x03#id\x12\x15\n\x13\n\x1113549061668422190\n\x16\n\x06number\x12\x0c\x1a\n\n\x08\xae\x84\xa5\xfe\xcb\xd9\x88\x18',
              ),
              (
                  b'12948957000457930',
                  b'\nL\n\x14\n\x06issuer\x12\n\n\x08\n\x06GDBank\n\x1c\n\x03#id\x12\x15\n\x13\n\x1112948957000457930\n\x16\n\x06number\x12\x0c\x1a\n\n\x08\xca\xbd\xe5\xf1\x9f\xa0\x80\x17',
              ),
              (
                  b'11163838768727470',
                  b'\nP\n\x16\n\x06number\x12\x0c\x1a\n\n\x08\xae\x8b\x8f\xa1\xbc\xae\xea\x13\n\x1c\n\x03#id\x12\x15\n\x13\n\x1111163838768727470\n\x18\n\x06issuer\x12\x0e\n\x0c\n\nBellsGarbo',
              ),
              (
                  b'11191576325053580',
                  b'\nK\n\x1c\n\x03#id\x12\x15\n\x13\n\x1111191576325053580\n\x16\n\x06number\x12\x0c\x1a\n\n\x08\x8c\xb9\xd3\xda\xde\xd5\xf0\x13\n\x13\n\x06issuer\x12\t\n\x07\n\x05BofBC',
              ),
              (
                  b'11290312140467510',
                  b'\nL\n\x14\n\x06issuer\x12\n\n\x08\n\x06GDBank\n\x16\n\x06number\x12\x0c\x1a\n\n\x08\xb6\xda\xa4\xa4\xaa\x8f\x87\x14\n\x1c\n\x03#id\x12\x15\n\x13\n\x1111290312140467510',
              ),
              (
                  b'11385846637304370',
                  b'\nP\n\x16\n\x06number\x12\x0c\x1a\n\n\x08\xb2\xb4\xa2\x82\xe0\xeb\x9c\x14\n\x18\n\x06issuer\x12\x0e\n\x0c\n\nBellsGarbo\n\x1c\n\x03#id\x12\x15\n\x13\n\x1111385846637304370',
              ),
              (
                  b'11470379189154620',
                  b'\nM\n\x15\n\x06issuer\x12\x0b\n\t\n\x07HeyBank\n\x16\n\x06number\x12\x0c\x1a\n\n\x08\xbc\xe6\x88\xa8\xfc\x87\xb0\x14\n\x1c\n\x03#id\x12\x15\n\x13\n\x1111470379189154620',
              ),
              (
                  b'11584989140147230',
                  b'\nK\n\x1c\n\x03#id\x12\x15\n\x13\n\x1111584989140147230\n\x16\n\x06number\x12\x0c\x1a\n\n\x08\x9e\xb8\xb3\xd3\xc7\x8f\xca\x14\n\x13\n\x06issuer\x12\t\n\x07\n\x05BofBC',
              ),
              (
                  b'11739198589848540',
                  b'\nL\n\x14\n\x06issuer\x12\n\n\x08\n\x06GDBank\n\x16\n\x06number\x12\x0c\x1a\n\n\x08\xdc\x97\x95\xcf\xd2\x97\xed\x14\n\x1c\n\x03#id\x12\x15\n\x13\n\x1111739198589848540',
              ),
              (
                  b'11771673810809530',
                  b'\nP\n\x1c\n\x03#id\x12\x15\n\x13\n\x1111771673810809530\n\x16\n\x06number\x12\x0c\x1a\n\n\x08\xba\xe5\x9e\x9f\xe6\xc8\xf4\x14\n\x18\n\x06issuer\x12\x0e\n\x0c\n\nBellsGarbo',
              ),
              (
                  b'12441028369470600',
                  b'\nK\n\x16\n\x06number\x12\x0c\x1a\n\n\x08\x88\xa9\xe9\xa4\xca\xe1\x8c\x16\n\x13\n\x06issuer\x12\t\n\x07\n\x05BofBC\n\x1c\n\x03#id\x12\x15\n\x13\n\x1112441028369470600',
              ),
              (
                  b'12968701241275060',
                  b'\nP\n\x18\n\x06issuer\x12\x0e\n\x0c\n\nBellsGarbo\n\x1c\n\x03#id\x12\x15\n\x13\n\x1112968701241275060\n\x16\n\x06number\x12\x0c\x1a\n\n\x08\xb4\xe5\xbc\xf5\xf0\xde\x84\x17',
              ),
              (
                  b'12982257258547830',
                  b'\nP\n\x18\n\x06issuer\x12\x0e\n\x0c\n\nBellsGarbo\n\x16\n\x06number\x12\x0c\x1a\n\n\x08\xf6\x94\x9d\x82\xb5\xe9\x87\x17\n\x1c\n\x03#id\x12\x15\n\x13\n\x1112982257258547830',
              ),
              (
                  b'13019350102369400',
                  b'\nP\n\x18\n\x06issuer\x12\x0e\n\x0c\n\nBellsGarbo\n\x1c\n\x03#id\x12\x15\n\x13\n\x1113019350102369400\n\x16\n\x06number\x12\x0c\x1a\n\n\x08\xf8\xc8\xbb\xd0\xfa\xa0\x90\x17',
              ),
              (
                  b'13916484476264770',
                  b'\nL\n\x14\n\x06issuer\x12\n\n\x08\n\x06GDBank\n\x16\n\x06number\x12\x0c\x1a\n\n\x08\xc2\xc2\xeb\xcd\x80\x9f\xdc\x18\n\x1c\n\x03#id\x12\x15\n\x13\n\x1113916484476264770',
              ),
              (
                  b'14453480592564160',
                  b'\nK\n\x13\n\x06issuer\x12\t\n\x07\n\x05BofBC\n\x16\n\x06number\x12\x0c\x1a\n\n\x08\xc0\xbf\xf3\x83\xd3\xab\xd6\x19\n\x1c\n\x03#id\x12\x15\n\x13\n\x1114453480592564160',
              ),
              (
                  b'14844931107602160',
                  b'\nP\n\x18\n\x06issuer\x12\x0e\n\x0c\n\nBellsGarbo\n\x16\n\x06number\x12\x0c\x1a\n\n\x08\xf0\xfd\x88\xeb\xad\xac\xaf\x1a\n\x1c\n\x03#id\x12\x15\n\x13\n\x1114844931107602160',
              ),
              (
                  b'14912408563871390',
                  b'\nK\n\x1c\n\x03#id\x12\x15\n\x13\n\x1114912408563871390\n\x16\n\x06number\x12\x0c\x1a\n\n\x08\x9e\xdd\xc7\xf9\x9a\xd8\xbe\x1a\n\x13\n\x06issuer\x12\t\n\x07\n\x05BofBC',
              ),
              (
                  b'14990890937985390',
                  b'\nP\n\x18\n\x06issuer\x12\x0e\n\x0c\n\nBellsGarbo\n\x1c\n\x03#id\x12\x15\n\x13\n\x1114990890937985390\n\x16\n\x06number\x12\x0c\x1a\n\n\x08\xee\xd2\xe6\xc9\xac\xc4\xd0\x1a',
              ),
              (
                  b'15054318664602640',
                  b'\nM\n\x15\n\x06issuer\x12\x0b\n\t\n\x07HeyBank\n\x1c\n\x03#id\x12\x15\n\x13\n\x1115054318664602640\n\x16\n\x06number\x12\x0c\x1a\n\n\x08\x90\xa8\xdb\xa2\xab\xfa\xde\x1a',
              ),
              (
                  b'16011471358128450',
                  b'\nP\n\x1c\n\x03#id\x12\x15\n\x13\n\x1116011471358128450\n\x18\n\x06issuer\x12\x0e\n\x0c\n\nBellsGarbo\n\x16\n\x06number\x12\x0c\x1a\n\n\x08\xc2\xe2\x87\xf5\x92\xcb\xb8\x1c',
              ),
              (
                  b'16073125141142750',
                  b'\nL\n\x1c\n\x03#id\x12\x15\n\x13\n\x1116073125141142750\n\x14\n\x06issuer\x12\n\n\x08\n\x06GDBank\n\x16\n\x06number\x12\x0c\x1a\n\n\x08\xde\xb9\xdf\x93\xc1\xcd\xc6\x1c',
              ),
              (
                  b'16283233487191600',
                  b'\nP\n\x18\n\x06issuer\x12\x0e\n\x0c\n\nBellsGarbo\n\x1c\n\x03#id\x12\x15\n\x13\n\x1116283233487191600\n\x16\n\x06number\x12\x0c\x1a\n\n\x08\xb0\xfc\xb1\xde\xbb\xb0\xf6\x1c',
              ),
              (
                  b'16827485386298040',
                  b'\nP\n\x1c\n\x03#id\x12\x15\n\x13\n\x1116827485386298040\n\x16\n\x06number\x12\x0c\x1a\n\n\x08\xb8\xa5\xa7\x87\xa4\x90\xf2\x1d\n\x18\n\x06issuer\x12\x0e\n\x0c\n\nBellsGarbo',
              ),
              (
                  b'17035680063294790',
                  b'\nP\n\x18\n\x06issuer\x12\x0e\n\x0c\n\nBellsGarbo\n\x16\n\x06number\x12\x0c\x1a\n\n\x08\xc6\x9a\xb8\xd5\xc5\xbb\xa1\x1e\n\x1c\n\x03#id\x12\x15\n\x13\n\x1117035680063294790',
              ),
              (
                  b'17396883707513070',
                  b'\nP\n\x18\n\x06issuer\x12\x0e\n\x0c\n\nBellsGarbo\n\x16\n\x06number\x12\x0c\x1a\n\n\x08\xee\x91\xd7\x8c\xfa\xcb\xf3\x1e\n\x1c\n\x03#id\x12\x15\n\x13\n\x1117396883707513070',
              ),
              (
                  b'17861046738135650',
                  b'\nM\n\x16\n\x06number\x12\x0c\x1a\n\n\x08\xe2\xd4\x92\x91\xf0\x90\xdd\x1f\n\x1c\n\x03#id\x12\x15\n\x13\n\x1117861046738135650\n\x15\n\x06issuer\x12\x0b\n\t\n\x07HeyBank',
              ),
              (
                  b'18362223127059380',
                  (
                      b'\nL\n\x16\n\x06number\x12\x0c\x1a\n\n\x08\xb4\xb7\x99\xd6\x83\x8b\xcf'
                      b' \n\x1c\n\x03#id\x12\x15\n\x13\n\x1118362223127059380\n\x14\n\x06issuer\x12\n\n\x08\n\x06GDBank'
                  ),
              ),
              (
                  b'18526138896540830',
                  (
                      b'\nL\n\x1c\n\x03#id\x12\x15\n\x13\n\x1118526138896540830\n\x14\n\x06issuer\x12\n\n\x08\n\x06GDBank\n\x16\n\x06number\x12\x0c\x1a\n\n\x08\x9e\xc9\xf3\xbf\xcd\xad\xf4 '
                  ),
              ),
              (
                  b'18569067217418250',
                  (
                      b'\nL\n\x14\n\x06issuer\x12\n\n\x08\n\x06GDBank\n\x1c\n\x03#id\x12\x15\n\x13\n\x1118569067217418250\n\x16\n\x06number\x12\x0c\x1a\n\n\x08\x8a\xf0\xb7\xfa\xfd\x8e\xfe '
                  ),
              ),
              (
                  b'18878522895102380',
                  b'\nM\n\x1c\n\x03#id\x12\x15\n\x13\n\x1118878522895102380\n\x15\n\x06issuer\x12\x0b\n\t\n\x07HeyBank\n\x16\n\x06number\x12\x0c\x1a\n\n\x08\xac\xe3\xaf\x98\xaa\xbd\xc4!',
              ),
              (
                  b'18889177882781580',
                  b'\nK\n\x1c\n\x03#id\x12\x15\n\x13\n\x1118889177882781580\n\x16\n\x06number\x12\x0c\x1a\n\n\x08\x8c\xcf\xb5\x8e\xb7\xf3\xc6!\n\x13\n\x06issuer\x12\t\n\x07\n\x05BofBC',
              ),
              (
                  b'19991040399813050',
                  b'\nK\n\x16\n\x06number\x12\x0c\x1a\n\n\x08\xba\xd3\xe2\xed\xec\xb7\xc1#\n\x1c\n\x03#id\x12\x15\n\x13\n\x1119991040399813050\n\x13\n\x06issuer\x12\t\n\x07\n\x05BofBC',
              ),
          ],
          'edges/owns_card': [
              (b'1876448', b'16827485386298040'),
              (b'1372437', b'11470379189154620'),
              (b'1368305', b'11163838768727470'),
              (b'1974494', b'16011471358128450'),
              (b'1257724', b'18569067217418250'),
              (b'1758057', b'17396883707513070'),
              (b'1531660', b'14844931107602160'),
              (b'1489311', b'1238474857489384'),
              (b'1407706', b'11290312140467510'),
              (b'196838', b'17861046738135650'),
              (b'1195675', b'8878522895102384'),
              (b'1659366', b'13019350102369400'),
              (b'1499004', b'11470379189154620'),
              (b'1344333', b'16283233487191600'),
              (b'1443888', b'9991040399813057'),
              (b'1108778', b'14912408563871390'),
              (b'175583', b'11290312140467510'),
              (b'1251872', b'12948957000457930'),
              (b'1493851', b'3549061668422198'),
              (b'1599418', b'9991040399813057'),
              (b'1768701', b'18362223127059380'),
              (b'1549489', b'1238474857489384'),
              (b'1879799', b'18569067217418250'),
              (b'125454', b'18526138896540830'),
          ],
      }
      converted = root | unigraph_utils.ReadAndConvertUnigraph(
          schema, self.resource_dir
      )
      converted_customers = (converted['nodes/customer']
                             | beam.MapTuple(
                                 lambda x, y: (x, _tf_example_from_bytes(y))))
      converted_creditcard = (converted['nodes/creditcard']
                              | beam.MapTuple(
                                  lambda x, y: (x, _tf_example_from_bytes(y))))
      util.assert_that(
          converted_customers,
          util.equal_to(
              [
                  (id, _tf_example_from_bytes(ex))
                  for id, ex in expected_dict['nodes/customer']
              ]
          ),
          label='assert_customers',
      )
      util.assert_that(
          converted_creditcard,
          util.equal_to(
              [
                  (id, _tf_example_from_bytes(ex))
                  for id, ex in expected_dict['nodes/creditcard']
              ]
          ),
          label='assert_creditcard',
      )
      util.assert_that(
          converted['edges/owns_card'],
          util.equal_to(expected_dict['edges/owns_card']),
          label='assert_owns_card')
      root.run()


# This function is needed because serialization to bytes in Python
# is non-deterministic
def _tf_example_from_bytes(s: bytes):
  ex = tf.train.Example()
  ex.ParseFromString(s)
  return ex

if __name__ == '__main__':
  tf.test.main()
