% step function

function output = activation_for_logistics_step_f(z)
    if z >= 0.5
        output = 1; % versicolor
    else
        output = 0; % setosa
    end
   
end