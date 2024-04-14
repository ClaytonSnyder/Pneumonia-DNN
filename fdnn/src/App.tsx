import { BrowserRouter, Route, Routes, Link as RouterLink, LinkProps as RouterLinkProps } from "react-router-dom";
import { ThemeProvider, createTheme } from "@mui/material/styles";
import "./App.css";
import { Layout } from "./components/Layout";
import { DatasetPage } from "./components/DatasetPage";
import { ProjectsPage } from "./components/ProjectsPage";
import * as React from "react";
import { LinkProps } from "@mui/material/Link";

const LinkBehavior = React.forwardRef<HTMLAnchorElement, Omit<RouterLinkProps, "to"> & { href: RouterLinkProps["to"] }>(
    (props, ref) => {
        const { href, ...other } = props;
        // Map href (Material UI) -> to (react-router)
        return <RouterLink ref={ref} to={href} {...other} />;
    }
);

const theme = createTheme({
    components: {
        MuiLink: {
            defaultProps: {
                component: LinkBehavior,
            } as LinkProps,
        },
        MuiButtonBase: {
            defaultProps: {
                LinkComponent: LinkBehavior,
            },
        },
    },
});

function App() {
    return (
        <ThemeProvider theme={theme}>
            <BrowserRouter>
                <Routes>
                    <Route path="/" element={<Layout />}>
                        <Route path="/" element={<DatasetPage />} />
                        <Route path="/datasets" element={<DatasetPage />} />
                        <Route path="/projects" element={<ProjectsPage />} />
                    </Route>
                </Routes>
            </BrowserRouter>
        </ThemeProvider>
    );
}

export default App;
