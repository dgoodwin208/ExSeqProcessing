function varargout = colorCorrection_manual(varargin)
% COLORCORRECTION_MANUAL MATLAB code for colorCorrection_manual.fig
%      COLORCORRECTION_MANUAL, by itself, creates a new COLORCORRECTION_MANUAL or raises the existing
%      singleton*.
%
%      H = COLORCORRECTION_MANUAL returns the handle to a new COLORCORRECTION_MANUAL or the handle to
%      the existing singleton*.
%
%      COLORCORRECTION_MANUAL('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in COLORCORRECTION_MANUAL.M with the given input arguments.
%
%      COLORCORRECTION_MANUAL('Property','Value',...) creates a new COLORCORRECTION_MANUAL or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before colorCorrection_manual_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to colorCorrection_manual_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help colorCorrection_manual

% Last Modified by GUIDE v2.5 09-May-2018 09:17:13

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
    'gui_Singleton',  gui_Singleton, ...
    'gui_OpeningFcn', @colorCorrection_manual_OpeningFcn, ...
    'gui_OutputFcn',  @colorCorrection_manual_OutputFcn, ...
    'gui_LayoutFcn',  [] , ...
    'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT

function setMontage(handles)
mov_bounds_y = handles.montagecrop(1):handles.montagecrop(1)+handles.montagecrop(3)-1;
mov_bounds_x = handles.montagecrop(2):handles.montagecrop(2)+handles.montagecrop(4)-1;

mov_translated_img = imtranslate3D(handles.mov_img,round(handles.offsets_current));

imgToShow = imfuse(handles.ref_img(mov_bounds_y,mov_bounds_x,handles.ref_zval),...
    mov_translated_img(mov_bounds_y,mov_bounds_x,handles.ref_zval),...
    'falsecolor','Scaling','independent');
imshow(imgToShow,'Parent',handles.axes_montageimg);

set(handles.text_offsetDisplay,'string',sprintf('%i,%i,%i',...
    handles.offsets_current(1),handles.offsets_current(2),handles.offsets_current(3)));

function updatedHandles = loadImages(handles)
loadParameters;
handles.params = params;


set(handles.edit_directorybox,'string',handles.params.deconvolutionImagesDir);

load(fullfile(params.colorCorrectionImagesDir,...
    sprintf('%s-downsample_round%.03i_colorcalcs.mat',params.FILE_BASENAME,handles.roundnum)));

handles.offsets_chan01 = chan2_offsets;
handles.offsets_chan02 = chan3_offsets;
handles.offsets_chan03 = chan4_offsets;
handles.offsets_current = chan2_offsets;



handles.ref_img = load3DTif_uint16(fullfile(handles.params.deconvolutionImagesDir,...
    sprintf('%s-downsample_round%.03i_ch00.tif',handles.params.FILE_BASENAME,handles.roundnum)));

handles.mov_img = load3DTif_uint16(fullfile(handles.params.deconvolutionImagesDir,...
    sprintf('%s-downsample_round%.03i_ch0%i.tif',handles.params.FILE_BASENAME,handles.roundnum,handles.movChan )));

handles.ref_zval = floor(size(handles.ref_img,3)/2);

set(handles.edit_roundnum,'string',num2str(handles.roundnum));

imagesc(handles.axes_refimg,handles.ref_img(:,:,handles.ref_zval) );
set(handles.axes_refimg,'xtick',[],'ytick',[])

updatedHandles = handles;


% --- Executes just before colorCorrection_manual is made visible.
function colorCorrection_manual_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to colorCorrection_manual (see VARARGIN)

handles.roundnum = 2;
handles.movChan = 1;
handles = loadImages(handles);

%Initialize somethings
set(handles.slider_refimg,'Min',1);
set(handles.slider_refimg,'Value',floor(size(handles.ref_img,3)/2));
set(handles.slider_refimg,'Max',size(handles.ref_img,3));
set(handles.text_refimgzmin,'string','1');
set(handles.text_refimgmaxz,'string',sprintf('%i',size(handles.ref_img,3)));

handles.montagecrop = [1,1,size(handles.ref_img,1),size(handles.ref_img,2)];
xl = get(handles.axes_refimg,'Xlim');
yl = get(handles.axes_refimg,'Ylim');
handles.axes_refimg_xlim = xl;
handles.axes_refimg_ylim = yl;

% Choose default command line output for colorCorrection_manual
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

setMontage(handles);
% UIWAIT makes colorCorrection_manual wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = colorCorrection_manual_OutputFcn(hObject, eventdata, handles)
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in pushbutton_startROI.
function pushbutton_startROI_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_startROI (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


%Remove the previous ROI box
if isfield(handles,'axes_refimg_roishape')
    delete(handles.axes_refimg_roishape);
end
[x,y] = ginput(2);
x= floor(x); y= floor(y);

%Draw the rectangular region
top_left = [y(1),x(1)];
height = diff(y);
width = diff(x);

%Funky re-organzing of rectangular coordinates to deal with MATLAB's XY
%shifting for imshow
handles.axes_refimg_roishape = rectangle('Position',[top_left(2), top_left(1), width, height],'EdgeColor','r');
handles.montagecrop = [top_left(1), top_left(2), height,width];

guidata(hObject,handles); %Save the data into the handles construct
setMontage(handles);

% --- Executes on selection change in popupmenu1.
function popupmenu1_Callback(hObject, eventdata, handles)
% hObject    handle to popupmenu1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

contents = cellstr(get(hObject,'String'));
channel_choice = contents{get(hObject,'Value')};

if strcmp(channel_choice,'ch01')
    handles.offsets_current= handles.offsets_chan01;
    handles.movChan = 1;
    handles.mov_img = load3DTif_uint16(fullfile(handles.params.deconvolutionImagesDir,...
        sprintf('%s-downsample_round%.03i_ch01.tif',handles.params.FILE_BASENAME,handles.roundnum)));
elseif strcmp(channel_choice,'ch02')
    handles.offsets_current= handles.offsets_chan02;
    handles.movChan = 2;
    handles.mov_img = load3DTif_uint16(fullfile(handles.params.deconvolutionImagesDir,...
        sprintf('%s-downsample_round%.03i_ch02.tif',handles.params.FILE_BASENAME,handles.roundnum)));
elseif strcmp(channel_choice,'ch03')
    handles.offsets_current= handles.offsets_chan03;
    handles.movChan = 3;
    handles.mov_img = load3DTif_uint16(fullfile(handles.params.deconvolutionImagesDir,...
        sprintf('%s-downsample_round%.03i_ch03.tif',handles.params.FILE_BASENAME,handles.roundnum)));
else
    fprintf('I did the logic wrong');
end


guidata(hObject,handles);
setMontage(handles);



% --- Executes during object creation, after setting all properties.
function popupmenu1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to popupmenu1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit_roundnum_Callback(hObject, eventdata, handles)
% hObject    handle to edit_roundnum (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit_roundnum as text
%        str2double(get(hObject,'String')) returns contents of edit_roundnum as a double

handles.roundnum = str2double(get(hObject,'String'));
handles = loadImages(handles);
guidata(hObject,handles);
setMontage(handles)

% --- Executes during object creation, after setting all properties.
function edit_roundnum_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit_roundnum (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on slider movement.
function slider_refimg_Callback(hObject, eventdata, handles)
% hObject    handle to slider_refimg (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider

z = round(get(hObject,'Value'));
handles.ref_zval = z;

guidata(hObject,handles);

imagesc(handles.axes_refimg,handles.ref_img(:,:,handles.ref_zval) );
set(handles.axes_refimg,'xtick',[],'ytick',[])

set(handles.axes_refimg,'Xlim',handles.axes_refimg_xlim);
set(handles.axes_refimg,'Ylim',handles.axes_refimg_ylim);
setMontage(handles);




% --- Executes during object creation, after setting all properties.
function slider_refimg_CreateFcn(hObject, eventdata, handles)
% hObject    handle to slider_refimg (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end


% --- Executes on slider movement.
function slider_montageimg_Callback(hObject, eventdata, handles)
% hObject    handle to slider_montageimg (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider


% --- Executes during object creation, after setting all properties.
function slider_montageimg_CreateFcn(hObject, eventdata, handles)
% hObject    handle to slider_montageimg (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end



function edit_directorybox_Callback(hObject, eventdata, handles)
% hObject    handle to edit_directorybox (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit_directorybox as text
%        str2double(get(hObject,'String')) returns contents of edit_directorybox as a double


% --- Executes during object creation, after setting all properties.
function edit_directorybox_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit_directorybox (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on key press with focus on edit_directorybox and none of its controls.
function edit_directorybox_KeyPressFcn(hObject, eventdata, handles)
% hObject    handle to edit_directorybox (see GCBO)
% eventdata  structure with the following fields (see MATLAB.UI.CONTROL.UICONTROL)
%	Key: name of the key that was pressed, in lower case
%	Character: character interpretation of the key(s) that was pressed
%	Modifier: name(s) of the modifier key(s) (i.e., control, shift) pressed
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on mouse press over axes background.
function axes_refimg_ButtonDownFcn(hObject, eventdata, handles)
% hObject    handle to axes_refimg (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in pushbutton3.
function pushbutton3_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
figure;
imagesc(handles.ref_img(:,:,handles.ref_zval));
title('Use zoom tools, press ENTER when done');
pause;
xl = xlim;
yl = ylim;
handles.axes_refimg_xlim = xl;
handles.axes_refimg_ylim = yl;
guidata(hObject,handles);
close;
set(handles.axes_refimg,'Xlim',handles.axes_refimg_xlim);
set(handles.axes_refimg,'Ylim',handles.axes_refimg_ylim);


% --- Executes on key press with focus on figure1 and none of its controls.
function figure1_KeyPressFcn(hObject, eventdata, handles)
% hObject    handle to figure1 (see GCBO)
% eventdata  structure with the following fields (see MATLAB.UI.FIGURE)
%	Key: name of the key that was pressed, in lower case
%	Character: character interpretation of the key(s) that was pressed
%	Modifier: name(s) of the modifier key(s) (i.e., control, shift) pressed
% handles    structure with handles and user data (see GUIDATA)
switch eventdata.Key
    case 'rightarrow'
        handles.offsets_current = handles.offsets_current + [0 1 0];
    case 'leftarrow'
        handles.offsets_current = handles.offsets_current + [0 -1 0];
    case 'uparrow'
        handles.offsets_current = handles.offsets_current + [-1 0 0];
    case 'downarrow'
        handles.offsets_current = handles.offsets_current + [1 0 0];
    case 'a'
        handles.offsets_current = handles.offsets_current + [0 0 1];
    case 's'
        handles.offsets_current = handles.offsets_current + [0 0 -1];
end

switch handles.movChan
    case 1
        handles.offsets_chan01 = handles.offsets_current;
    case 2
        handles.offsets_chan02 = handles.offsets_current;
    case 3
        handles.offsets_chan03 = handles.offsets_current;
end
guidata(hObject,handles);
setMontage(handles);


% --- Executes on button press in pushbutton_save.
function pushbutton_save_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_save (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

chan2_offsets = handles.offsets_chan01;
chan3_offsets = handles.offsets_chan02;
chan4_offsets = handles.offsets_chan03;

save(fullfile(handles.params.colorCorrectionImagesDir,sprintf('%s-downsample_round%.03i_colorcalcs.mat',...
    handles.params.FILE_BASENAME,handles.roundnum)),...
    'chan2_offsets',...
    'chan3_offsets',...
    'chan4_offsets');
fprintf('Saved the colorcalcs.mat file\n');
