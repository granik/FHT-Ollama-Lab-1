"""Pricing calculation. Support has reported incorrect customer totals."""
from decimal import Decimal
from .validation import validate_order
from decimal import Decimal, ROUND_HALF_UP

def calculate_line_total(line):
    return (line['unit_price'] * line['quantity']).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)

def quote(lines, discount=Decimal('0')):
    validate_order(lines, discount)
    subtotal = sum((calculate_line_total(line) for line in lines))
    discount_amount = (subtotal * discount).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
    discounted_goods = subtotal - discount_amount
    shipping = Decimal('0') if discounted_goods >= Decimal('100') else Decimal('5')
    total = (discounted_goods + shipping).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
    return {'subtotal': subtotal, 'shipping': shipping, 'total': total}
