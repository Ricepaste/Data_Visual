function EBPTA1

load IRIS_IN.csv;
load IRIS_OUT.csv;
input = IRIS_IN;
target = IRIS_OUT;

% initialize the weight matrix
outputmatrix=zeros(12,1);
for i=1:1:12
 for j=1:1:1
   outputmatrix(i,j)=rand;
 end
end

hiddenmatrix=zeros(4,12);
for i=1:1:4
 for j=1:1:12
   hiddenmatrix(i,j)=rand;
 end
end


RMSE =zeros(100,1);


% Training
for epoch=1:1:100
    t=[];

    for iter=1:1:75


    
    
    
         doutputnet=dpurelin(SUMout);
         deltaoutput=(target(iter)-Aout)*doutputnet;
         error=target(iter)-Aout;
         t=[t;error.^2];

    end
    RMSE(epoch) = sqrt(sum(t)/75);
    fprintf('epoch %.0f:  RMSE = %.3f\n',epoch, sqrt(sum(t)/75));
end

%======================================================
fprintf('\nTotal number of epochs: %g\n', epoch);
fprintf('Final RMSE: %g\n', RMSE(epoch));
plot(1:epoch,RMSE(1:epoch));
legend('Training');
ylabel('RMSE');xlabel('Epoch');


Tot_Correct=0;

for i=76:length(input)
    
  
    
        if Aout > target(i)-0.5 && Aout <= target(i)+0.5
            Tot_Correct=Tot_Correct+ 1;
        end
end    
Tot_Percent= (Tot_Correct) / (length(input)-75);
Test_correct_percent=Tot_Percent