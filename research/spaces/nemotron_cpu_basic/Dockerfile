FROM debian:bookworm-slim@sha256:7c7b2c966bc9ee8cedfeef67e0e279108992c77681fa595db4a9d65c06ccc587

# Container-only packages; no Mac/Homebrew/Python environment changes.
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl libstdc++6 libgomp1 \
    && rm -rf /var/lib/apt/lists/*
RUN curl --fail --location --proto '=https' --tlsv1.2 --max-time 120 \
      https://github.com/NVIDIA/NeMo-Speech.cpp/releases/download/v0.2.0/nemo-speech-0.2.0-linux-x86_64-cpu.tar.gz -o /tmp/runtime.tar.gz \
    && echo 'f396057150f1b774935c7414fd32ebc04195a795a34120d78d8c9ecefa1b7507  /tmp/runtime.tar.gz' | sha256sum -c - \
    && mkdir -p /opt/nemo && tar -xzf /tmp/runtime.tar.gz --strip-components=1 -C /opt/nemo \
    && rm /tmp/runtime.tar.gz
RUN mkdir -p /opt/models \
    && curl --fail --location --proto '=https' --tlsv1.2 --max-time 600 \
      https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b/resolve/1c8deaecc64b91f034d73e08dd8b64625eb3395d/nemotron-3.5-asr-streaming-0.6b.q8_0.gguf \
      -o /opt/models/nemotron.gguf \
    && echo 'a5c435f294eea8f88ce68dd27b8c3bfea7f777cb2fbba04fcd30eaa555f429ae  /opt/models/nemotron.gguf' | sha256sum -c -
RUN useradd --create-home --uid 1000 app
COPY --chmod=755 start.sh /opt/start.sh
USER app
ENV HOME=/home/app OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
EXPOSE 7860
CMD ["/opt/start.sh"]
