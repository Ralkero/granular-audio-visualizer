using NAudio.Wave;
using NAudio.CoreAudioApi;
using System.Threading.Channels;

// Stdout is a binary PCM pipe only, never a recording file. Stderr carries status.
// Modes: "<PID>" captures a process tree; "system <endpoint-id|default>" captures a render
// endpoint in shared-mode loopback (read-only; opens no render stream, changes no routing).
uint pid = 0;
var system = args.Length == 2 && args[0] == "system";
if (!system && (args.Length != 1 || !uint.TryParse(args[0], out pid)))
{
    Console.Error.WriteLine("Usage: ProcessLoopback <PID> | ProcessLoopback system <endpoint-id|default>");
    return 2;
}
if (!OperatingSystem.IsWindowsVersionAtLeast(10, 0, 20348))
{
    Console.Error.WriteLine("Process loopback requires Windows build 20348 or newer.");
    return 3;
}
try
{
    using var cancel = new CancellationTokenSource();
    var packets = Channel.CreateBounded<Packet>(new BoundedChannelOptions(16) {
        FullMode = BoundedChannelFullMode.DropOldest, SingleReader = true, SingleWriter = true
    });
    using var enumerator = new MMDeviceEnumerator();
    using var endpoint = !system ? null
        : args[1] == "default" ? enumerator.GetDefaultAudioEndpoint(DataFlow.Render, Role.Multimedia)
        : enumerator.GetDevice(args[1]);
    // System mode keeps the endpoint mix format so no conversion hides the device layout.
    await using var recorder = system
        ? await new WasapiRecorderBuilder()
            .WithDevice(endpoint!)
            .WithLoopbackCapture()
            .WithBufferLength(20)
            .BuildAsync()
        : await new WasapiRecorderBuilder()
            .WithProcessLoopback(pid, ProcessLoopbackMode.IncludeTargetProcessTree)
            .WithFormat(WaveFormat.CreateIeeeFloatWaveFormat(48000, 2))
            .WithBufferLength(20)
            .BuildAsync();
    var format = recorder.WaveFormat;
    var isFloat = format.Encoding == WaveFormatEncoding.IeeeFloat
        || (format is WaveFormatExtensible extensible && extensible.SubFormat == AudioMediaSubtypes.MEDIASUBTYPE_IEEE_FLOAT);
    if (format.BitsPerSample != 32 || !isFloat)
    {
        Console.Error.WriteLine($"ERROR: Unsupported capture format: {format}");
        return 4;
    }
    int rate = format.SampleRate, channels = format.Channels;
    long sequence = 0;
    Exception? captureError = null;
    recorder.DataAvailable += (buffer, flags, position, qpc) => {
        // Copy while WASAPI owns this span; bounded transfer, never pipe I/O or DSP here.
        if (buffer.Length == 0) return; // Event wake-ups without data carry no samples or timestamp.
        packets.Writer.TryWrite(new Packet(buffer.ToArray(), Interlocked.Increment(ref sequence), qpc, (uint)flags));
    };
    recorder.RecordingStopped += (_, e) => {
        captureError = e.Exception;
        packets.Writer.TryComplete();
        cancel.Cancel();
    };
    _ = Task.Run(async () => {
        await Console.In.ReadLineAsync(); // "stop" or parent pipe closure
        cancel.Cancel();
    });
    recorder.StartRecording();
    Console.Error.WriteLine(system
        ? $"READY: system loopback, {rate} Hz, {channels} channels float32"
        : "READY: process-tree loopback, 48000 Hz, stereo float32");
    using var output = new BinaryWriter(Console.OpenStandardOutput());
    try
    {
        await foreach (var packet in packets.Reader.ReadAllAsync(cancel.Token))
        {
            output.Write(new byte[] { 71, 77, 86, 49 }); // GMV1
            output.Write(packet.Data.Length);
            output.Write(rate);
            output.Write(channels);
            output.Write(packet.Sequence);
            output.Write(packet.Qpc);
            output.Write(packet.Flags);
            output.Write(packet.Data);
            output.Flush();
        }
    }
    catch (OperationCanceledException) { }
    recorder.StopRecording();
    if (captureError != null) throw captureError;
    return 0;
}
catch (Exception e)
{
    Console.Error.WriteLine($"ERROR: {e.GetType().Name}: {e.Message}");
    return 1;
}

record Packet(byte[] Data, long Sequence, long Qpc, uint Flags);
