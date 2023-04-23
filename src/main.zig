const layer = @import("layer.zig");
const nll = @import("nll.zig");
const mnist = @import("mnist.zig");
const relu = @import("relu.zig");

const std = @import("std");

const INPUT_SIZE: u32 = 784;
const OUTPUT_SIZE: u32 = 10;
const BATCH_SIZE: u32 = 32;
const EPOCHS: u32 = 50;

pub fn main() !void {

    // Get MNIST data
    const mnist_data = try mnist.read_mnist(&allocator);

    // Prep loss function
    const loss_function = nll.NLL(OUTPUT_SIZE);

    // Prep NN
    var layer1 = try layer.Layer(INPUT_SIZE, 100).init(&allocator);
    var relu1 = relu.Relu.new();
    var layer2 = try layer.Layer(100, OUTPUT_SIZE).init(&allocator);

    // Do training
    var e: usize = 0;
    while (e < EPOCHS) : (e += 1) {
        // Do training
        var i: usize = 0;
        while (i < 60000 / BATCH_SIZE) : (i += 1) {
            // Prep inputs and targets
            const inputs = mnist_data.train_images[i * INPUT_SIZE * BATCH_SIZE .. (i + 1) * INPUT_SIZE * BATCH_SIZE];
            const targets = mnist_data.train_labels[i * BATCH_SIZE .. (i + 1) * BATCH_SIZE];

            // Go forward and get loss
            const outputs1 = try layer1.forward(inputs, &allocator);
            const outputs2 = try relu1.forward(outputs1, &allocator);
            const outputs3 = try layer2.forward(outputs2, &allocator);
            const loss = try loss_function.nll(outputs3, targets, &allocator);

            // Update network
            const grads1 = try layer2.backwards(loss.input_grads, &allocator);
            const grads2 = try relu1.backwards(grads1.input_grads, &allocator);
            const grads3 = try layer1.backwards(grads2, &allocator);
            layer1.apply_gradients(grads3.weight_grads);
            layer2.apply_gradients(grads1.weight_grads);

            // Free memory
            allocator.free(outputs1);
            allocator.free(outputs2);
            allocator.free(outputs3);
            allocator.free(grads1.weight_grads);
            allocator.free(grads1.input_grads);
            allocator.free(grads2);
            allocator.free(grads3.weight_grads);
            allocator.free(grads3.input_grads);
            allocator.free(loss.loss);
            allocator.free(loss.input_grads);
        }

        // Do validation
        i = 0;
        var correct: f64 = 0;
        const outputs1 = try layer1.forward(mnist_data.test_images, &allocator);
        const outputs2 = try relu1.forward(outputs1, &allocator);
        const outputs3 = try layer2.forward(outputs2, &allocator);
        var b: usize = 0;
        while (b < 10000) : (b += 1) {
            var max_guess: f64 = outputs3[b * OUTPUT_SIZE];
            var guess_index: usize = 0;
            for (outputs3[b * OUTPUT_SIZE .. (b + 1) * OUTPUT_SIZE]) |o, oi| {
                if (o > max_guess) {
                    max_guess = o;
                    guess_index = oi;
                }
            }
            if (guess_index == mnist_data.test_labels[b]) {
                correct += 1;
            }
        }

        // Free memory
        allocator.free(outputs1);
        allocator.free(outputs2);
        allocator.free(outputs3);

        correct = correct / 10000;
        std.debug.print("Average Validation Accuracy: {}\n", .{correct});
    }

    layer1.destruct(&allocator);
    mnist_data.destruct(&allocator);
}

test "Forward once" {
    var allocator = std.testing.allocator;

    const loss_function = nll.NLL(2);

    // Create layer with custom weights
    var layer1 = try layer.Layer(2, 2).init(&allocator);
    allocator.free(layer1.weights);
    var custom_weights = [4]f64{ 0.1, 0.2, 0.3, 0.4 };
    layer1.weights = &custom_weights;

    // Test forward pass outputs
    var inputs_array = [_]f64{ 0.1, 0.2, 0.3, 0.4 };
    const inputs: []f64 = &inputs_array;
    const outputs = try layer1.forward(inputs, &allocator);
    const expected_outputs = [4]f64{
        0.07,
        0.1,
        0.15,
        0.22,
    };
    var i: u32 = 0;
    while (i < 4) : (i += 1) {
        try std.testing.expectApproxEqRel(expected_outputs[i], outputs[i], 0.000000001);
    }

    // Test loss outputs
    var targets_array = [_]u8{ 0, 1 };
    const targets: []u8 = &targets_array;
    const loss = try loss_function.nll(outputs, targets, &allocator);
    allocator.free(outputs);
    const expected_loss = [2]f64{ 0.7082596763414484, 0.658759555548697 };
    i = 0;
    while (i < 2) : (i += 1) {
        try std.testing.expectApproxEqRel(loss.loss[i], expected_loss[i], 0.000000001);
    }

    // Test loss input_grads
    const expected_loss_input_grads = [4]f64{
        -5.074994375506203e-01,
        5.074994375506204e-01,
        4.8250714233361025e-01,
        -4.8250714233361025e-01,
    };
    i = 0;
    while (i < 4) : (i += 1) {
        try std.testing.expectApproxEqRel(loss.input_grads[i], expected_loss_input_grads[i], 0.000000001);
    }

    // Do layer backwards
    const grads = try layer1.backwards(loss.input_grads, &allocator);

    // Test layer weight grads
    const expected_layer_weight_grads = [4]f64{
        4.700109947251052e-02,
        -4.7001099472510514e-02,
        4.575148471166002e-02,
        -4.5751484711660004e-02,
    };
    i = 0;
    while (i < 4) : (i += 1) {
        try std.testing.expectApproxEqRel(expected_layer_weight_grads[i], grads.weight_grads[i], 0.000000001);
    }

    // Test layer input grads
    const expected_layer_input_grads = [4]f64{
        5.074994375506206e-02,
        5.074994375506209e-02,
        -4.8250714233361025e-02,
        -4.8250714233361025e-02,
    };
    i = 0;
    while (i < 4) : (i += 1) {
        try std.testing.expectApproxEqRel(grads.input_grads[i], expected_layer_input_grads[i], 0.000000001);
    }

    allocator.free(grads.weight_grads);
    allocator.free(grads.input_grads);

    allocator.free(loss.loss);
    allocator.free(loss.input_grads);
}
