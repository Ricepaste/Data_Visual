load IRIS_IN.csv;
load IRIS_OUT.csv;
input = IRIS_IN;
learning = 0.45;

num_epochs = 100;

NumNeuron = 12;

target = zeros(150, 3);

for i = 1:1:150
    if (IRIS_OUT(i) == 1)
        target(i, 1) = 1;
    elseif (IRIS_OUT(i) == 2)
        target(i, 2) = 1;
    else
        target(i, 3) = 1;
    end
end

% initialize the weight matrix
outputmatrix = zeros(NumNeuron, 3);
for i = 1:NumNeuron
    for j = 1:3
        outputmatrix(i, j) = rand;
    end
end

hiddenmatrix = zeros(4, NumNeuron);
for i = 1:4
    for j = 1:NumNeuron
        hiddenmatrix(i, j) = rand;
    end
end

RMSE = zeros(num_epochs, 1);

% Training
for epoch = 1:num_epochs
    t = [];

    for iter = 1:75
        % Forward pass
        SUMhid = input(iter, :) * hiddenmatrix;
        Ahid = logsig(SUMhid);
        SUMout = Ahid * outputmatrix;
        Aout = softmax(SUMout.T);

        C = mod(iter, 3);
        if (C == 0)
            possibility = Aout(3, 1);
            C = 3;
        elseif (C == 1)
            possibility = Aout(1, 1);
        else 
            possibility = Aout(2, 1);
        end

        crossentropy = -log(1 - possibility);

        % Backpropagation
        DELTAout = (target(iter, :) - SUMout);
        dTRANSFERout = dpurelin(SUMout);
        error = (target(iter, :)-SUMout);
        t = [t; error(C).^2];

        % Weight updates
        DELTAhid = DELTAout .* dTRANSFERout .* outputmatrix;
        dTRANSFERhid = dlogsig(Ahid, logsig(SUMhid));

        outputmatrix = outputmatrix + 0.45 * DELTAout .* dTRANSFERout .* Ahid.T;

        for i = 1:1:12
            hiddenmatrix(:, i) = hiddenmatrix(:, i) + 0.45 * DELTAhid(i) .* dTRANSFERhid(i) .* input(i, :).T;
        end

    end

    RMSE(epoch) = sqrt(sum(t) / 75);
    fprintf('epoch %.0f:  RMSE = %.3f\n', epoch, sqrt(sum(t) / 75));
end

fprintf('\nTotal number of epochs: %g\n', epoch);
fprintf('Final RMSE: %g\n', RMSE(epoch));
plot(1:epoch, RMSE(1:epoch));
legend('Training');
ylabel('RMSE');
xlabel('Epoch');

Tot_Correct = 0;

for i = 76:length(input)
    SUMhid = input(i, :) * hiddenmatrix;
    Ahid = logsig(SUMhid);
    SUMout = Ahid * outputmatrix;
    Aout = softmax(SUMout')';

    C = mod(i, 3);
    if (C == 0)
        C = 3;
    elseif (C == 1)
        C = 1;
    else
        C = 2;
    end

    [~, predicted_class] = max(Aout);
    i
    predicted_class
    C
    
    if predicted_class == C
        Tot_Correct = Tot_Correct + 1;
    end
end
Tot_Percent = Tot_Correct / 75;
Tot_Correct
Test_correct_percent = Tot_Percent