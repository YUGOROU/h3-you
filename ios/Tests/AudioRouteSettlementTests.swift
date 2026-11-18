import Foundation
import Testing

private actor Samples {
    let values: [Bool]
    private(set) var count = 0
    init(_ values: [Bool]) { self.values = values }
    func next() -> Bool {
        let value = values[min(count, values.count - 1)]
        count += 1
        return value
    }
}

@Test func routeMayBecomeBuiltInAfterInitialHFP() async throws {
    let samples = Samples([false, false, true, true, true])
    let ready = try await AudioRouteSettlement.wait(sample: { await samples.next() }, pause: {})
    #expect(ready)
    #expect(await samples.count == 5)
}

@Test func persistentHFPHasBoundedTimeout() async throws {
    let samples = Samples([false])
    let ready = try await AudioRouteSettlement.wait(sample: { await samples.next() }, pause: {})
    #expect(!ready)
    #expect(await samples.count == 31)
}

@Test func transientBuiltInRouteDoesNotPassStabilityCheck() async throws {
    let samples = Samples([true, true, false, true, true, true])
    let ready = try await AudioRouteSettlement.wait(sample: { await samples.next() }, pause: {})
    #expect(ready)
    #expect(await samples.count == 6)
}

@Test @MainActor func cancelledStartDoesNotInspectRoute() async {
    let samples = Samples([true])
    let task = Task {
        try await AudioRouteSettlement.wait(sample: { await samples.next() }, pause: {})
    }
    task.cancel()
    do {
        _ = try await task.value
        Issue.record("Cancelled setup should not succeed")
    } catch {
        #expect(error is CancellationError)
    }
    #expect(await samples.count == 0)
}

@Test func cancellationDuringSetupExitsBeforeFurtherSamples() async {
    let samples = Samples([false])
    do {
        _ = try await AudioRouteSettlement.wait(sample: { await samples.next() }, pause: { throw CancellationError() })
        Issue.record("Cancelled setup should not succeed")
    } catch {
        #expect(error is CancellationError)
    }
    #expect(await samples.count == 1)
}

@Test func interruptionAbortsWithoutFurtherWaiting() async {
    enum Interruption: Error { case began }
    do {
        _ = try await AudioRouteSettlement.wait(sample: { throw Interruption.began }, pause: {
            Issue.record("Interrupted route must not wait again")
        })
        Issue.record("Interrupted setup should not succeed")
    } catch {
        #expect(error is Interruption)
    }
}
