import * as React from 'react';
import { render } from 'react-dom';
import Application from 'web-component/Applicaton';
import {ROOT_SELECTOR} from 'web-constant/selector';

const element = document.querySelector(ROOT_SELECTOR);

render(<Application />, element);
