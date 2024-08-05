import * as React from "react";
import AppBar from "@mui/material/AppBar";
import Box from "@mui/material/Box";
import Toolbar from "@mui/material/Toolbar";
import Typography from "@mui/material/Typography";
import IconButton from "@mui/material/IconButton";
import MenuIcon from "@mui/icons-material/Menu";
import Drawer from "@mui/material/Drawer";
import { Outlet } from "react-router-dom";
import MenuList from "@mui/material/MenuList";
import MenuItem from "@mui/material/MenuItem";
import ListItemIcon from "@mui/material/ListItemIcon";
import SendIcon from "@mui/icons-material/Send";
import PriorityHighIcon from "@mui/icons-material/PriorityHigh";
import { Link } from "@mui/material";

const drawerWidth = 240;

export function Layout() {
    const [open, setOpen] = React.useState(true);
    const handleDrawerToggle = () => {
        setOpen(!open);
    };

    return (
        <>
            <Box sx={{ flexGrow: 1 }}>
                <AppBar
                    position="static"
                    style={{
                        marginLeft: open ? drawerWidth : 0,
                    }}
                >
                    <Toolbar>
                        <IconButton
                            size="large"
                            edge="start"
                            color="inherit"
                            aria-label="menu"
                            onClick={handleDrawerToggle}
                            sx={{ mr: 2 }}
                        >
                            <MenuIcon />
                        </IconButton>
                        <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
                            Deep Learning - Image Classification Pipeline
                        </Typography>
                    </Toolbar>
                </AppBar>
            </Box>
            <Drawer
                sx={{
                    width: drawerWidth,
                    flexShrink: 0,
                    "& .MuiDrawer-paper": {
                        width: drawerWidth,
                        boxSizing: "border-box",
                    },
                }}
                variant="persistent"
                anchor="left"
                open={open}
            >
                <MenuList>
                    <MenuItem component={Link} href="/projects">
                        <ListItemIcon>
                            <SendIcon fontSize="small" />
                        </ListItemIcon>
                        <Typography variant="inherit">Projects</Typography>
                    </MenuItem>
                    <MenuItem component={Link} href="/datasets">
                        <ListItemIcon>
                            <PriorityHighIcon fontSize="small" />
                        </ListItemIcon>
                        <Typography variant="inherit">Datasets</Typography>
                    </MenuItem>
                </MenuList>
            </Drawer>
            <div
                style={{
                    marginLeft: open ? drawerWidth : 0,
                }}
            >
                <Outlet />
            </div>
        </>
    );
}
