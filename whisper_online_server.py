#!/usr/bin/env python3
from whisper_online import *
import sys
import argparse
import os
import logging
import numpy as np

logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser()
# server options
parser.add_argument("--host", type=str, default="localhost")
parser.add_argument("--port", type=int, default=43007)
parser.add_argument(
    "--warmup-file",
    type=str,
    dest="warmup_file",
    help="The path to a speech audio wav file to warm up Whisper so that the very first chunk processing is fast. It can be e.g. https://github.com/ggerganov/whisper.cpp/raw/master/samples/jfk.wav .",
)
# options from whisper_online
add_shared_args(parser)
args = parser.parse_args()
set_logging(args, logger, other="")
# setting whisper object by args
SAMPLING_RATE = 16000
size = args.model
language = args.lan
asr, online = asr_factory(args)
min_chunk = args.min_chunk_size
# warm up the ASR because the very first transcribe takes more time than the others.
# Test results in https://github.com/ufal/whisper_streaming/pull/81
msg = "Whisper is not warmed up. The first chunk processing may take longer."
if args.warmup_file:
    if os.path.isfile(args.warmup_file):
        a = load_audio_chunk(args.warmup_file, 0, 1)
        asr.transcribe(a)
        logger.info("Whisper is warmed up.")
    else:
        logger.critical("The warm up file is not available. " + msg)
        sys.exit(1)
else:
    logger.warning(msg)

######### Server objects
import asyncio
import websockets
import json
import io
import soundfile


class Connection:
    """it wraps websocket object"""

    PACKET_SIZE = 32000 * 5 * 60  # 5 minutes # was: 65536

    def __init__(self, websocket):
        self.websocket = websocket
        self.last_line = ""

    async def send(self, line):
        """it doesn't send the same line twice"""
        if line == self.last_line:
            return
        await self.websocket.send(line)
        self.last_line = line

    async def receive_audio(self):
        try:
            data = await self.websocket.recv()
            return data
        except websockets.ConnectionClosed:
            return None


class ServerProcessor:
    def __init__(self, conn, online_asr_proc, min_chunk):
        self.connection = conn
        self.online_asr_proc = online_asr_proc
        self.min_chunk = min_chunk
        self.last_end = None
        self.is_first = True

    async def receive_audio_chunk(self):
        out = []
        minlimit = self.min_chunk * SAMPLING_RATE
        total_len = 0
        while total_len < minlimit:
            raw_bytes = await self.connection.receive_audio()
            if not raw_bytes:
                break
            # Expect bytes (not str)
            if isinstance(raw_bytes, str):
                raw_bytes = raw_bytes.encode("latin1")
            sf = soundfile.SoundFile(
                io.BytesIO(raw_bytes),
                channels=1,
                endian="LITTLE",
                samplerate=SAMPLING_RATE,
                subtype="PCM_16",
                format="RAW",
            )
            audio, _ = librosa.load(sf, sr=SAMPLING_RATE, dtype=np.float32)
            out.append(audio)
            total_len += len(audio)
        if not out:
            return None
        conc = np.concatenate(out)
        if self.is_first and len(conc) < minlimit:
            return None
        self.is_first = False
        return conc

    def format_output_transcript(self, o):
        if o[0] is not None:
            beg, end = o[0] * 1000, o[1] * 1000
            if self.last_end is not None:
                beg = max(beg, self.last_end)
            self.last_end = end
            print("%1.0f %1.0f %s" % (beg, end, o[2]), flush=True, file=sys.stderr)
            return "%1.0f %1.0f %s" % (beg, end, o[2])
        else:
            logger.debug("No text in this segment")
            return None

    async def send_result(self, o, type):
        msg = self.format_output_transcript(o)
        if msg is not None:
            await self.connection.send(json.dumps({"type": type, "text": msg}))

    async def process(self):
        self.online_asr_proc.init()
        while True:
            a = await self.receive_audio_chunk()
            if a is None:
                break
            self.online_asr_proc.insert_audio_chunk(a)
            output = online.process_iter()
            try:
                await self.send_result(output, "complete")
                await self.send_result(online.get_incomplete(), "incomplete")
            except websockets.ConnectionClosed:
                logger.info("connection closed")
                break


async def handler(websocket) -> None:
    logger.info("Connected to client")
    connection = Connection(websocket)
    proc = ServerProcessor(connection, online, args.min_chunk_size)
    await proc.process()
    logger.info("Connection to client closed")


async def main():
    async with websockets.serve(handler, args.host, args.port):
        logger.info("Listening on" + str((args.host, args.port)))
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    asyncio.run(main())
