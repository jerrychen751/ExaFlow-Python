from __future__ import annotations

import socket
import struct
import threading
import time
from typing import Any, Callable

import numpy as np
import pytest

pytestmark = pytest.mark.gui

pv = pytest.importorskip("pyvista")
QtCore = pytest.importorskip("PySide6.QtCore")

from PIL import Image

from exaflow.gui.streaming.base import DataFormat, TransportProtocol
from exaflow.gui.streaming.client import StreamingClient
from exaflow.gui.streaming.serializers import ImageSerializer, PickleSerializer
from exaflow.gui.streaming.server import StreamingServer
from exaflow.gui.streaming.transports import TCPSocketTransport
from exaflow.config import Case
from exaflow.fields import TimeLevel, allocate_state
from exaflow.io.writers import StreamWriter
from exaflow.mpi.subdomain import Subdomain

WAIT_SECONDS = 10.0


@pytest.fixture(scope="session")
def qt_application() -> Any:
    """
    The one QCoreApplication this process may hold. Qt allows a single instance, so every test in this module shares this fixture.
    """

    return QtCore.QCoreApplication.instance() or QtCore.QCoreApplication([])


@pytest.fixture
def listening_server(qt_application: Any) -> Callable[[], tuple[StreamingServer, int, list[Any]]]:
    def start() -> tuple[StreamingServer, int, list[Any]]:
        server = StreamingServer(port=0)
        assert server.is_listening()
        received: list[Any] = []
        server.data_received.connect(received.append)
        return server, server.read_port(), received

    return start


def spin_until(qt_application: Any, is_ready: Callable[[], bool], seconds: float = WAIT_SECONDS) -> bool:
    """
    Drive the Qt event loop until `is_ready` reports True or `seconds` pass, and report which happened. The test owns the main thread, so nothing delivers a socket signal unless the test processes events itself.
    """

    deadline = time.monotonic() + seconds
    while not is_ready() and time.monotonic() < deadline:
        qt_application.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents, 20)
    return is_ready()


def test_a_pickled_payload_reads_back_as_the_object_that_produced_it() -> None:
    serializer = PickleSerializer()
    velocity = np.arange(4.0)

    restored = serializer.deserialize(serializer.serialize({"velocity": velocity, "step": 3}))

    assert restored["step"] == 3
    assert np.array_equal(restored["velocity"], velocity)


def test_an_image_reads_back_at_the_same_size_and_colour() -> None:
    serializer = ImageSerializer()
    frame = np.zeros((4, 3, 3), dtype=np.uint8)
    frame[1, 2] = (255, 0, 0)

    restored = serializer.deserialize(serializer.serialize(frame))

    assert restored.size == (3, 4)
    assert np.asarray(restored)[1, 2].tolist() == [255, 0, 0]


def test_an_image_serializes_from_a_pillow_object_too() -> None:
    serializer = ImageSerializer()
    frame = Image.new("RGB", (2, 2), (0, 128, 255))

    restored = serializer.deserialize(serializer.serialize(frame))

    assert np.asarray(restored)[0, 0].tolist() == [0, 128, 255]


def test_an_image_serializer_refuses_a_payload_that_is_not_a_picture() -> None:
    with pytest.raises(ValueError, match="Unsupported image type"):
        ImageSerializer().serialize("a file name")


@pytest.mark.parametrize(
    "arguments,message",
    [
        (dict(transport=TransportProtocol.WEBSOCKET), "transport websocket is not implemented"),
        (dict(format=DataFormat.RENDERED_IMAGE), "data format rendered_image is not implemented"),
    ],
)
def test_the_client_refuses_a_combination_that_is_not_implemented(
    arguments: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(NotImplementedError, match=message):
        StreamingClient(**arguments)


def test_the_client_refuses_to_send_an_image() -> None:
    client = StreamingClient(port=1)
    with pytest.raises(NotImplementedError, match="Image streaming is not implemented"):
        client.send_image(np.zeros((2, 2, 3), dtype=np.uint8))


def test_the_server_refuses_a_protocol_that_is_not_implemented() -> None:
    with pytest.raises(ValueError, match="not yet implemented"):
        StreamingServer(protocol=TransportProtocol.HTTP)


def test_the_transport_writes_an_eight_byte_length_before_the_payload() -> None:
    """
    The server reads the stream, not a datagram, so the prefix is the only mark that says where one message ends. It is a big-endian unsigned 64-bit integer, which is what struct '!Q' packs.
    """

    payload = b"a serialized dataset"
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    address, port = listener.getsockname()
    received: list[bytes] = []

    def accept() -> None:
        connection, _ = listener.accept()
        with connection:
            length = struct.unpack("!Q", connection.recv(8))[0]
            received.append(connection.recv(length))

    thread = threading.Thread(target=accept)
    thread.start()
    try:
        assert TCPSocketTransport(address, port).send(payload)
    finally:
        thread.join(WAIT_SECONDS)
        listener.close()

    assert received == [payload]


def test_the_transport_reports_a_refused_connection_rather_than_raising() -> None:
    """
    A run must not stop because the viewer is closed, so a failed send returns False and the caller marches on.
    """

    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    _, port = listener.getsockname()
    listener.close()

    assert not TCPSocketTransport("127.0.0.1", port, timeout=0.5).send(b"payload")


@pytest.mark.filterwarnings(
    "ignore:Setting the shape on a NumPy array has been deprecated:DeprecationWarning"
)
def test_a_dataset_reaches_the_server_as_the_dataset_that_was_sent(
    qt_application: Any,
    listening_server: Callable[[], tuple[StreamingServer, int, list[Any]]],
) -> None:
    server, port, received = listening_server()
    grid = pv.ImageData(dimensions=(3, 3, 1))
    grid.point_data["pressure"] = np.arange(9.0)
    sent: list[bool] = []

    thread = threading.Thread(target=lambda: sent.append(StreamingClient(port=port).send_dataset(grid)))
    thread.start()
    try:
        assert spin_until(qt_application, lambda: bool(received)), "the server received nothing"
    finally:
        thread.join(WAIT_SECONDS)

    assert sent == [True]
    assert np.asarray(received[0].point_data["pressure"]).tolist() == list(range(9))


def test_a_message_split_across_two_writes_is_reassembled(
    qt_application: Any,
    listening_server: Callable[[], tuple[StreamingServer, int, list[Any]]],
) -> None:
    """
    TCP carries a stream, so a payload arrives in whatever pieces the network hands over. The server holds the pieces until the byte count the prefix names is complete.
    """

    server, port, received = listening_server()
    payload = PickleSerializer().serialize({"step": 7})
    connection = socket.create_connection(("127.0.0.1", port), timeout=WAIT_SECONDS)
    try:
        connection.sendall(struct.pack("!Q", len(payload)))
        connection.sendall(payload[:5])
        assert not spin_until(qt_application, lambda: bool(received), 0.5)
        connection.sendall(payload[5:])
        assert spin_until(qt_application, lambda: bool(received)), "the server received nothing"
    finally:
        connection.close()

    assert received[0] == {"step": 7}


@pytest.mark.filterwarnings(
    "ignore:Setting the shape on a NumPy array has been deprecated:DeprecationWarning"
)
def test_the_stream_writer_sends_the_whole_domain_with_its_level(
    qt_application: Any,
    listening_server: Callable[[], tuple[StreamingServer, int, list[Any]]],
    build_case: Callable[..., Case],
    build_subdomain: Callable[..., Subdomain],
) -> None:
    """
    The grid that arrives is the one the viewer shows: a rectilinear grid in metres, with the pressure and the velocity as point data and the step and the time as field data, exactly as a `.vtr` file states them.
    """

    server, port, received = listening_server()
    case = build_case((4, 3))
    subdomain = build_subdomain(case.grid)
    state = allocate_state(subdomain, 1)
    state.pressure[subdomain.interior] = np.arange(12.0).reshape(4, 3)
    writer = StreamWriter(port, case.grid, subdomain, None, frequency=2)

    thread = threading.Thread(target=lambda: writer.write("Original", state, TimeLevel(4, 0.5, 0.125)))
    thread.start()
    try:
        assert spin_until(qt_application, lambda: bool(received)), "the server received nothing"
    finally:
        thread.join(WAIT_SECONDS)

    grid = received[0]
    assert isinstance(grid, pv.RectilinearGrid)
    assert grid.dimensions == (4, 3, 1)
    assert grid.bounds == (0.0, 1.0, 0.0, 1.0, 0.0, 0.0)
    assert np.asarray(grid.point_data["pressure"]).reshape(3, 4).T.tolist() == np.arange(12.0).reshape(4, 3).tolist()
    assert np.asarray(grid.point_data["velocity"]).shape == (12, 3)
    assert int(grid.field_data["StepIndex"][0]) == 4
    assert float(grid.field_data["TimeValue"][0]) == 0.5
    assert writer.frequency == 2


@pytest.mark.filterwarnings(
    "ignore:Setting the shape on a NumPy array has been deprecated:DeprecationWarning"
)
def test_the_stream_writer_marches_on_when_no_viewer_listens(
    build_case: Callable[..., Case],
    build_subdomain: Callable[..., Subdomain],
) -> None:
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    _, port = listener.getsockname()
    listener.close()
    case = build_case((4,))
    subdomain = build_subdomain(case.grid)

    StreamWriter(port, case.grid, subdomain, None).write("Original", allocate_state(subdomain, 1), TimeLevel(0, 0.0, 0.1))
