import * as React from 'react';
import { RouteComponentProps } from 'react-router-dom';
import * as Routes from 'web-constant/route';
import WebLayout from 'web-component/layout/WebLayout';

function ApplicationRoute({ match: { url } }) {
  const webLayout = (<WebLayout path={url} />);

  if(url === Routes.DEMO_ROUTE){
    return webLayout;
  }

}

export default ApplicationRoute;
