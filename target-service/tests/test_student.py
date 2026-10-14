import unittest
from decimal import Decimal as D
from orders.pricing import quote
from orders.service import place_order
from orders.storage import OrderStore

def lines(price, quantity=1):
    return [{'unit_price': D(price), 'quantity': quantity}]


class StudentRegression(unittest.TestCase):
    def test_regression(self):
        result = quote(lines("10", 2))
        self.assertEqual(result["subtotal"], D("20.00"))
        self.assertEqual(result["shipping"], D("5.00"))
        self.assertEqual(result["total"], D("25.00"))