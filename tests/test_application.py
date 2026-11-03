import io
import pytest
from gmv.sources.application import HEADER, read_packet


def test_native_packet_protocol_preserves_qpc():
    packet = HEADER.pack(b"GMV1", 16, 48000, 2, 5, 123456789, 1) + bytes(16)
    data, rate, channels, seq, qpc, flags = read_packet(io.BytesIO(packet))
    assert len(data) == 16 and rate == 48000 and channels == 2
    assert seq == 5 and qpc == 123456789 and flags == 1


@pytest.mark.parametrize("packet", [b"GMV", HEADER.pack(b"BAD!", 16, 48000, 2, 1, 0, 0),
                                   HEADER.pack(b"GMV1", 16, 48000, 2, 1, 0, 0) + bytes(8),
                                   HEADER.pack(b"GMV1", 2**30, 48000, 2, 1, 0, 0)])
def test_native_packet_rejects_bad_or_unbounded_data(packet):
    with pytest.raises(ValueError):
        read_packet(io.BytesIO(packet))
