using NAudio.Wave;
using NAudio.CoreAudioApi;
using System.Threading.Channels;

// Stdout is a binary PCM pipe only, never a recording file. Stderr carries status.
if (args.Length != 1 || !uint.TryParse(args[0], out var pid))
{
    Console.Error.WriteLine("Usage: ProcessLoopback <PID>");
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
    await using var recorder = await new WasapiRecorderBuilder()
        .WithProcessLoopback(pid, ProcessLoopbackMode.IncludeTargetProcessTree)
        .WithFormat(WaveFormat.CreateIeeeFloatWaveFormat(48000, 2))
        .WithBufferLength(20)
        .BuildAsync();
    long sequence = 0;
    Exception? captureError = null;
    recorder.DataAvailable += (buffer, flags, position, qpc) => {
        // Copy while WASAPI owns this span; bounded transfer, never pipe I/O or DSP here.
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
    Console.Error.WriteLine("READY: process-tree loopback, 48000 Hz, stereo float32");
    using var output = new BinaryWriter(Console.OpenStandardOutput());
    try
    {
        await foreach (var packet in packets.Reader.ReadAllAsync(cancel.Token))
        {
            output.Write(new byte[] { 71, 77, 86, 49 }); // GMV1
            output.Write(packet.Data.Length);
            output.Write(48000);
            output.Write(2);
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
