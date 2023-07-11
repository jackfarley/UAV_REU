

figure('Position', [200 200 600 600 ]);
FONTSIZE_NUMBERS = 10;
COLOR_NUMBERS_ONE = 'r';
COLOR_NUMBERS_TWO = 'b';
CIRCLE_RADIUS = 0.5;
hold on
grid on
xlabel('o_{max} = 1', 'FontSize',14)
ylabel('Y coordinate','FontSize',14)
gs = 5;
line([gs gs],[gs 0],'Color','black','LineStyle','-','LineWidth',2);
line([0 gs],[gs gs],'Color','black','LineStyle','-','LineWidth',2);
line([0 0],[gs 0],'Color','black','LineStyle','-','LineWidth',2);
line([0 gs],[0 0],'Color','black','LineStyle','-','LineWidth',2);
xticks(0:1:gs)
yticks(0:1:gs)


%#GBS were at 1,1, 20,1, 5,5, 10,10, and 15,5










x = [1,1; 2, 1; 3, 1; 4, 1];


text(1,1,'start',FontSize=10)
text(20,1, 'end', FontSize=10)
R_g = 1;
circle(1,1,R_g)
circle(0,0,R_g)
circle(1,3, R_g)
circle(3,1,R_g)
circle(3,3, R_g)
circle(0,4, R_g)


movingPoint_one = rectangle('Parent',gca,'Position',[0,0,0.5,0.5],'Curvature',[1,1],'FaceColor','b');

for frame=1:length(x)
set(movingPoint_one,'Position',[x(frame,1),x(frame,2),.75,.75])
frames(frame)=getframe;
pause(.4)
end
'''
(1, 1, 'valid')
1/1 [==============================] - 0s 25ms/step
(2, 1, 'valid')
1/1 [==============================] - 0s 26ms/step
(3, 1, 'valid')
1/1 [==============================] - 0s 23ms/step
(4, 1, 'valid')
'[''